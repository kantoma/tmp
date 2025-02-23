import math
import numpy as np
import pandas as pd
from scipy.special import comb, beta
from scipy.optimize import minimize_scalar

class DirichletModel:
    """
    NBD-Dirichletモデルに関するパラメータ(K, S)の推定と、ブランド別指標の計算を担うクラス。

    【主な役割】
      1) K, S を入力データから推定（通常のコンストラクタ __init__ ）
      2) すでに K, S, M が判明している場合、推定をスキップしてインスタンスを生成（from_parameters）
      3) ブランド別ペネトレーション, 理論的購入回数, カテゴリ平均購入回数などを計算
    """

    def __init__(
        self,
        cat_pen,          # カテゴリ全体のペネトレーション (例: カテゴリ購入者数 / 全体母数)
        cat_buyrate,      # カテゴリバイヤー1人あたりの平均購入回数
        brand_share,      # 各ブランドの市場シェア (合計1.0 程度が望ましい)
        brand_pen_obs,    # 観測されたブランド別ペネトレーション (推定後の比較などに用いる)
        brand_name=None,  # ブランド名のリスト (省略時は B1, B2,...)
        cat_pur_var=None, # カテゴリ購入率の「分散」が既知の場合に指定 (K推定を省略するため)
        nstar=50,         # NBD分布を計算する際の購入回数の上限 (0..nstar)
        max_S=30,         # Sパラメータを探索する際の最大値
        max_K=30,         # Kパラメータを探索する際の最大値
        check=False,      # 推定過程のデバッグログを表示するかどうか
        skip_estimation=False  # TrueにするとK, Sの推定を行わない (from_parameters で利用)
    ):
        """
        通常のコンストラクタ:
          - cat_pen, cat_buyrate などから K, S を推定 (skip_estimation=Falseの場合)
          - skip_estimation=True なら推定ルーチンをスキップ (ユーザが後からK, Sを手動セット)
        """

        # 入力パラメータをインスタンス変数に格納
        self.cat_pen = cat_pen         # カテゴリペネトレーション
        self.cat_buyrate = cat_buyrate # カテゴリバイヤー平均購入回数
        self.brand_share = brand_share # ブランドシェア
        self.brand_pen_obs = brand_pen_obs # 観測されたブランドペネトレーション
        self.nstar = nstar            # NBDを0~nstarまで計算
        self.max_S = max_S            # Sパラメータ最大探索値
        self.max_K = max_K            # Kパラメータ最大探索値
        self.check = check            # デバッグログを出すかどうか

        # ブランド数
        self.nbrand = len(brand_pen_obs)

        # ブランド名が与えられなければ B1..Bn の連番とする
        if brand_name is None:
            self.brand_name = [f"B{i+1}" for i in range(self.nbrand)]
        else:
            self.brand_name = brand_name

        # 基準期間(Period=1)での平均購入回数 = cat_pen * cat_buyrate
        self.M0 = self.cat_pen * self.cat_buyrate
        # 現在の M (解析対象期間の平均購入回数)。初期は M0 と同じ
        self.M = self.M0

        # 識別用にクラス名を記録 (任意)
        self.class_name = "dirichlet"
        self.error = 0  # nstar のカバレッジチェックで使用

        if skip_estimation:
            # 推定をスキップする場合はK,Sはユーザが後からセットする想定なのでダミーを入れる
            self.K = None
            self.S = None
        else:
            # 通常: K と S の推定を実施
            self.K = self._estimate_K(cat_pur_var)
            self.S = self._estimate_S()
            # nstarが十分大きいかどうかチェック
            self._check_nstar_coverage()

    @classmethod
    def from_parameters(
        cls,
        S,          # 既知の Dirichletパラメータ S
        K,          # 既知の NBDパラメータ K
        M,          # カテゴリ全体の平均購入回数 (=cat_pen*cat_buyrateに相当)
        brand_share,# ブランドシェア
        brand_name=None,
        nstar=50,
        check=False
    ):
        """
        すでに S, K, M, brand_share がわかっている場合に、
        推定をスキップしてインスタンスを生成するためのクラスメソッド。

        - cat_pen などの推定用パラメータは使わないため、ダミーを設定して __init__ を呼ぶ
        - skip_estimation=True でK,S推定をブロックし、後からK,S,Mを上書き
        """
        # cat_pen=0.0 など適当なダミー値を入れておく
        # brand_pen_obs も 0埋めする (実際には使用しない)
        cat_pen_dummy = 0.0
        cat_buyrate_dummy = M
        brand_pen_obs_dummy = [0.0]*len(brand_share)

        # skip_estimation=True で初期化
        instance = cls(
            cat_pen=cat_pen_dummy,
            cat_buyrate=cat_buyrate_dummy,
            brand_share=brand_share,
            brand_pen_obs=brand_pen_obs_dummy,
            brand_name=brand_name,
            cat_pur_var=None,
            nstar=nstar,
            max_S=30,
            max_K=30,
            check=check,
            skip_estimation=True
        )
        # 推定をスキップした代わりに、ユーザ指定の S,K,M をそのままセット
        instance.S = S
        instance.K = K
        instance.M0 = M  # 基準期間の平均購入回数 (M0)
        instance.M = M   # 現在のM も M に合わせる

        return instance

    def _check_nstar_coverage(self):
        """
        nstar がカテゴリ分布(NBD)を十分にカバーできているか簡易チェックする。
        - 0..nstar までの合計が99%未満なら分布が切れている恐れあり
        - 分布の平均が self.M0 と大きくズレていたら要注意
        """
        prob_sum = sum(self._Pn(i) for i in range(self.nstar+1))
        mean_cat = sum(i * self._Pn(i) for i in range(self.nstar+1))
        if prob_sum < 0.99 or abs(mean_cat - self.M0) > 0.1:
            if self.check:
                print(f"[Warning] nstar={self.nstar} が分布を十分にカバーしていない可能性があります。")
            self.error = 1

    def _estimate_K(self, cat_pur_var=None):
        """
        NBDパラメータ K を推定する内部メソッド。
        - cat_pur_var=None の場合 -> (ゼロ購買ロジック + 平均購入回数)から最適化
        - cat_pur_var を与える -> 分散式から K を計算
        """
        if cat_pur_var is None:
            # cat_pen からログを計算
            cp = math.log(1 - self.cat_pen)
            def eq1(k):
                # (k * log(1 + M/k) + log(1 - cat_pen))^2 を最小化
                return (k * math.log(1 + self.M / k) + cp) ** 2
            # 範囲 (0.0001, max_K) で探索
            res = minimize_scalar(eq1, bounds=(0.0001, self.max_K), method='bounded')
            return res.x
        else:
            # cat_pur_var が与えられたら M^2 / (cat_pur_var - M) の公式で直接計算
            return (self.M**2) / (cat_pur_var - self.M)

    def _estimate_S(self):
        """
        Dirichletパラメータ S を推定する内部メソッド。
        各ブランドについて理論ペネトレーションと観測値の誤差最小化 -> 外れ値除外 -> シェア加重平均
        """
        if self.check:
            print("[Info] 各ブランドに対してSを推定中...")

        def find_opt_S_for_brand(j):
            # ブランド j に対する最適Sを bounded検索
            res = minimize_scalar(self._eq2, bounds=(0, self.max_S), method='bounded', args=(j,))
            if self.check:
                print(f"[Debug] Brand {j+1}: S={res.x:.4f}, Objective={res.fun:.6f}")
            return res.x

        # 各ブランドごとに S を探索
        S_list = [find_opt_S_for_brand(j) for j in range(self.nbrand)]
        # 箱ひげ図から外れ値を判定
        bp = self._boxplot_stats(S_list)
        outlier = bp["out"]
        outlier2 = [s for s in S_list if s > bp["conf"][1]]
        outliers = outlier + outlier2

        # 外れ値を除いたものだけでシェア加重平均
        schoose = [(s not in outliers) for s in S_list]
        if self.check and len(outliers) > 0:
            idx_out_brands = [i+1 for i, sc in enumerate(schoose) if not sc]
            print(f"[Info] 外れ値を除外: {outliers}, 対象ブランド: {idx_out_brands}")

        chosen_S = [S_list[i] for i, sc in enumerate(schoose) if sc]
        chosen_w = [self.brand_share[i] for i, sc in enumerate(schoose) if sc]
        if sum(chosen_w) == 0:
            # 全部外れ値になった場合は 平均値で代替
            S_final = np.mean(S_list)
        else:
            S_final = np.average(chosen_S, weights=chosen_w)
        return S_final

    def _boxplot_stats(self, data):
        """
        箱ひげ図で使われる外れ値判定およびノッチ(conf区間)を計算する内部メソッド
        """
        arr = np.array(data)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        median = np.median(arr)
        n = len(arr)

        # 1.5IQRを超えるものを外れ値とする
        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr
        outliers = arr[(arr < lower_whisker) | (arr > upper_whisker)]

        # ノッチは median ± 1.58 * IQR / sqrt(n)
        conf_lower = median - 1.58 * iqr / math.sqrt(n)
        conf_upper = median + 1.58 * iqr / math.sqrt(n)

        return {
            "out": outliers.tolist(),
            "conf": [conf_lower, conf_upper]
        }

    def _eq2(self, S, j):
        """
        ブランドjの理論ペネトレーションと観測ペネトレーションの二乗誤差
        """
        t_pen = 1.0 - sum(self._Pn(i) * self._pzeron(i, j, S) for i in range(self.nstar+1))
        o_pen = self.brand_pen_obs[j]
        return (t_pen - o_pen) ** 2

    def _pzeron(self, n, j, S):
        """
        n回カテゴリを購入した人が、ブランドjを1回も買わない確率
        """
        alpha_j = S * self.brand_share[j]
        if n == 0:
            return 1.0
        return math.exp(sum(
            math.log(S - alpha_j + i) - math.log(S + i)
            for i in range(n)
        ))

    def _p_rj_n(self, rj, n, j, S):
        """
        n回カテゴリを購入した人が、ブランドjをrj回買う確率
        """
        alpha_j = S * self.brand_share[j]
        return comb(n, rj) * beta(alpha_j + rj, (S - alpha_j) + (n - rj)) / beta(alpha_j, (S - alpha_j))

    def _Pn(self, n):
        """
        カテゴリ全体のNBD分布 (カテゴリをn回買う確率) を計算
        """
        if n == 0:
            g = 0.0
        else:
            g = sum(math.log(self.K + a) - math.log(1 + a) for a in range(n))
        return math.exp(
            -self.K * math.log(1 + self.M / self.K)
            + g
            + n * math.log(self.M / (self.M + self.K))
        )

    def period_set(self, t):
        """
        解析対象期間を t倍に伸ばす (M0を t倍にするイメージ)
        """
        self.M = self.M0 * t

    def period_print(self):
        """
        現在の M が基準M0 の何倍かを表示
        """
        ratio = self.M / self.M0
        print(f"期間倍率: {round(ratio,2)}, 現在のM = {self.M}")

    def brand_pen(self, j):
        """
        ブランドjの理論的ペネトレーションを計算
        """
        p0 = sum(
            self._Pn(i) * self._p_rj_n(0, i, j, self.S)
            for i in range(self.nstar+1)
        )
        return 1.0 - p0

    def brand_buyrate(self, j):
        """
        ブランドjバイヤー1人あたりの理論購入回数
        """
        denom = self.brand_pen(j)
        if denom == 0:
            return 0.0

        def buyrate_n(n):
            return sum(
                r * self._p_rj_n(r, n, j, self.S)
                for r in range(1, n+1)
            )
        numerator = sum(
            self._Pn(n) * buyrate_n(n)
            for n in range(1, self.nstar+1)
        )
        return numerator / denom

    def wp(self, j):
        """
        ブランドjの購入者がカテゴリを合計何回買っているかの平均
        """
        denom = self.brand_pen(j)
        if denom == 0:
            return 0.0
        numerator = 0.0
        for n in range(1, self.nstar+1):
            numerator += n * self._Pn(n) * (1.0 - self._p_rj_n(0, n, j, self.S))
        return numerator / denom

    def get_brand_metrics(self):
        """
        ブランド別の指標をまとめ、pandas.DataFrame で返す
          - 観測ペネトレーション(Pen_Observed)
          - 理論ペネトレーション(Pen_Theoretical)
          - その差(Pen_Diff)
          - 理論的購入回数(BuyRate_Theoretical)
          - カテゴリ平均購入回数(WP_Theoretical)
        """
        records = []
        for j in range(self.nbrand):
            obs_pen = self.brand_pen_obs[j]
            thr_pen = self.brand_pen(j)
            diff_pen = thr_pen - obs_pen
            thr_buy = self.brand_buyrate(j)
            thr_wp = self.wp(j)
            records.append({
                "Brand": self.brand_name[j],
                "Share": self.brand_share[j],
                "Pen_Observed": obs_pen,
                "Pen_Theoretical": thr_pen,
                "Pen_Diff": diff_pen,
                "BuyRate_Theoretical": thr_buy,
                "WP_Theoretical": thr_wp
            })
        return pd.DataFrame(records)

    def get_parameters_summary(self):
        """
        現時点の推定/設定パラメータをDataFrameにして返す (S, K, Mなど)
        """
        data = {
            "S": [self.S],
            "K": [self.K],
            "M0": [self.M0],
            "M_current": [self.M],
            "cat_pen": [self.cat_pen],
            "cat_buyrate": [self.cat_buyrate],
        }
        return pd.DataFrame(data)


class DirichletSummary:
    """
    DirichletModel オブジェクトから各種サマリー (buy, freq, heavy, dup) を作成するクラス。
    Rの summary.dirichlet 相当。
    """

    def __init__(self, model: DirichletModel):
        """
        model : DirichletModel のインスタンス
        """
        self.model = model

    def summary_dirichlet(
        self,
        t=1,
        type=("buy","freq","heavy","dup"),
        digits=2,
        freq_cutoff=5,
        heavy_limit=None,
        dup_brand=1
    ):
        """
        指定した種類のサマリーをまとめて計算し、dict形式で返す。
        t : 集計期間倍率。model.period_set(t) によって M をスケールさせる
        type : ("buy", "freq", "heavy", "dup") などを指定 (複数可)
        digits : 小数点以下の丸め桁数
        freq_cutoff : freqサマリーでの上限（freq_cutoff+1回以上をまとめる）
        heavy_limit : heavyサマリーで「ヘビーバイヤー」とみなすカテゴリ購入回数 (デフォルト range(1,7))
        dup_brand : dupサマリーで重複購買を解析する際の注目ブランド(1-basedインデックス)
        """
        # 期間を t倍にして集計
        self.model.period_set(t)

        result = {}
        if heavy_limit is None:
            heavy_limit = range(1,7)

        def _brand_pen_with_limit(j, limit):
            p0 = 0.0
            for n in limit:
                p0 += self.model._Pn(n) * self.model._p_rj_n(0, n, j, self.model.S)
            return 1.0 - p0

        def _brand_buyrate_with_limit(j, limit):
            num = 0.0
            den = 0.0
            for n in limit:
                Pn_val = self.model._Pn(n)
                p_not0 = 1.0 - self.model._p_rj_n(0, n, j, self.model.S)
                sum_r = sum(r * self.model._p_rj_n(r, n, j, self.model.S) for r in range(1, n+1))
                num += Pn_val * sum_r
                den += Pn_val * p_not0
            if den == 0:
                return 0.0
            return num / den

        nbrand = self.model.nbrand
        brand_idxs = range(nbrand)

        for tt in type:
            if tt == "buy":
                # ブランドごとのペネトレーション(pen.brand)、購入率(pur.brand)、カテゴリ平均(pur.cat)を計算
                pen_brand = [self.model.brand_pen(j) for j in brand_idxs]
                pur_brand = [self.model.brand_buyrate(j) for j in brand_idxs]
                pur_cat   = [self.model.wp(j) for j in brand_idxs]
                df_buy = pd.DataFrame({
                    "pen.brand": pen_brand,
                    "pur.brand": pur_brand,
                    "pur.cat":   pur_cat
                }, index=self.model.brand_name).round(digits)
                result["buy"] = df_buy

            elif tt == "freq":
                # 各ブランドの購入回数分布を 0..freq_cutoff, freq_cutoff+1+ でまとめる
                def prob_r(r, j):
                    s = 0.0
                    for n in range(r, self.model.nstar+1):
                        s += self.model._Pn(n) * self.model._p_rj_n(r, n, j, self.model.S)
                    return s

                arr = np.zeros((nbrand, freq_cutoff+2))
                for j in brand_idxs:
                    for r_ in range(freq_cutoff+1):
                        arr[j, r_] = prob_r(r_, j)
                    tail_sum = 0.0
                    for n in range(freq_cutoff+1, self.model.nstar+1):
                        tail_sum += prob_r(n, j)
                    arr[j, freq_cutoff+1] = tail_sum

                col_names = list(range(freq_cutoff+1)) + [f"{freq_cutoff+1}+"]
                df_freq = pd.DataFrame(arr, index=self.model.brand_name, columns=col_names).round(digits)
                result["freq"] = df_freq

            elif tt == "heavy":
                # heavy_limit の範囲(例:1..6)内のカテゴリ購入者に限ったとき、
                # ブランドのペネトレーションと購入頻度を求める
                Pn_sum = sum(self.model._Pn(n) for n in heavy_limit)
                mat = np.zeros((nbrand, 2))
                for j in brand_idxs:
                    p0 = 0.0
                    for n in heavy_limit:
                        p0 += self.model._Pn(n)* self.model._p_rj_n(0, n, j, self.model.S)
                    pen_heavy = 1 - (p0 / Pn_sum)

                    buyrate_hlimit = _brand_buyrate_with_limit(j, heavy_limit)
                    brand_pen_full = self.model.brand_pen(j)
                    denom = Pn_sum - p0
                    if denom == 0:
                        avg_freq = 0.0
                    else:
                        avg_freq = buyrate_hlimit * brand_pen_full / denom
                    mat[j, 0] = pen_heavy
                    mat[j, 1] = avg_freq

                df_heavy = pd.DataFrame(
                    mat,
                    index=self.model.brand_name,
                    columns=["Penetration", "Avg Purchase Freq"]
                ).round(digits)
                result["heavy"] = df_heavy

            elif tt == "dup":
                # dup_brand で指定したブランド(1-based)をフォーカルブランドとして、
                # そのブランド購買者が他ブランドをどの程度重複購入しているか
                k_idx = dup_brand - 1
                r_dup = np.zeros(nbrand)
                r_dup[k_idx] = 1.0
                b_k = self.model.brand_pen(k_idx)
                other_idx = [x for x in brand_idxs if x != k_idx]

                for j in other_idx:
                    p0 = 0.0
                    for n in range(self.model.nstar+1):
                        alpha_sum = self.model.S*(self.model.brand_share[k_idx] + self.model.brand_share[j])
                        num = beta(alpha_sum, (self.model.S - alpha_sum) + n)
                        den = beta(alpha_sum, (self.model.S - alpha_sum))
                        p_comp_0 = num / den
                        p0 += self.model._Pn(n)* p_comp_0

                    b_jk = 1.0 - p0
                    b_j = self.model.brand_pen(j)
                    b_jk_both = b_j + b_k - b_jk
                    if b_k > 0:
                        b_j_given_k = b_jk_both / b_k
                    else:
                        b_j_given_k = 0.0
                    r_dup[j] = b_j_given_k

                s_dup = pd.Series(r_dup, index=self.model.brand_name).round(digits)
                result["dup"] = s_dup

        return result


# このファイルを直接実行した場合に、下記のデモコードが動作する
if __name__ == "__main__":
    # 【デモ１】: cat_penなどから自動推定するパターン
    cat_pen_example = 0.6
    cat_buyrate_example = 2.0
    brand_share_example = [0.3, 0.2, 0.5]
    brand_pen_obs_example = [0.15, 0.10, 0.35]
    brand_name_example = ["BrandA", "BrandB", "BrandC"]

    # 推定モードでモデルを作成
    model = DirichletModel(
        cat_pen=cat_pen_example,
        cat_buyrate=cat_buyrate_example,
        brand_share=brand_share_example,
        brand_pen_obs=brand_pen_obs_example,
        brand_name=brand_name_example,
        check=True
    )

    # サマリー作成クラスを用いて集計
    reporter = DirichletSummary(model)
    summaries = reporter.summary_dirichlet(
        t=1,
        type=("buy","freq","heavy","dup"),
        digits=2,
        freq_cutoff=5,
        heavy_limit=range(1,7),
        dup_brand=1
    )
    print("=== Summary buy ===")
    print(summaries["buy"], "\n")

    df_params = model.get_parameters_summary()
    print("=== Estimated Params ===")
    print(df_params, "\n")

    # 【デモ２】: S,K,M がすでにわかっている場合 (推定をスキップ)
    S_known = 5.0
    K_known = 3.0
    M_known = 1.2
    brand_share_known = [0.25, 0.19, 0.1, 0.1, 0.09, 0.08, 0.03, 0.02]

    model_known = DirichletModel.from_parameters(
        S=S_known,
        K=K_known,
        M=M_known,
        brand_share=brand_share_known,
        nstar=50,
        check=False
    )
    print("=== from_parameters: no estimation ===")
    print(model_known.get_parameters_summary(), "\n")

    # 簡単な確認: Brand0 (インデックス0) の理論ペネトレーション
    pen_b0 = model_known.brand_pen(0)
    print(f"Brand0 Pen (theoretical): {pen_b0}")
