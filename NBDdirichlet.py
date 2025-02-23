import math
import numpy as np
import pandas as pd
from scipy.special import comb, beta
from scipy.optimize import minimize_scalar

class DirichletModel:
    """
    DirichletモデルとNBDモデルのパラメータ推定および指標算出を行うクラス。
    
    【概要】
    - NBD(Negative Binomial Distribution) を用いてカテゴリ全体の購買頻度分布を表現
    - Dirichletモデルを用いて各ブランドへの購買割り振り（ペネトレーションや購入率）を説明
    - K (NBDのパラメータ) と S (Dirichletのパラメータ) をデータから推定
    - 推定されたパラメータをもとに、ブランド別の理論値(ペネトレーション、購入率など)を計算
    """

    def __init__(
        self,
        cat_pen,         # カテゴリ全体のペネトレーション（例えば、人口に対するそのカテゴリ購入者の割合）
        cat_buyrate,     # カテゴリバイヤーの平均購入回数（期間内での平均購入頻度）
        brand_share,     # 各ブランドの市場シェア（ブランド同士の比率; 合計1.0程度）
        brand_pen_obs,   # 観測されたブランドペネトレーション（実測データ）
        brand_name=None, # ブランド名のリスト（省略時は自動生成）
        cat_pur_var=None,# カテゴリ購入率の分散を既知として与える場合に使用（省略可）
        nstar=50,        # NBD分布を打ち切る上限値（カテゴリ購入回数を0～nstarまで考える）
        max_S=30,        # Sを探す際の最大探索値
        max_K=30,        # Kを探す際の最大探索値
        check=False      # デバッグメッセージを表示するかどうか
    ):
        """
        コンストラクタ：
        - 引数としてカテゴリペネトレーションやブランドシェア等を受け取り、
          内部でNBD-Dirichletモデルのパラメータ(K, S)を推定する。
        - 推定に使用するヘルパー関数（_estimate_K, _estimate_Sなど）を呼び出して初期化する。
        """
        self.cat_pen = cat_pen
        self.cat_buyrate = cat_buyrate
        self.brand_share = brand_share
        self.brand_pen_obs = brand_pen_obs
        self.nstar = nstar
        self.max_S = max_S
        self.max_K = max_K
        self.check = check

        # ブランド数
        self.nbrand = len(brand_pen_obs)

        # ブランド名が指定されていない場合は "B1", "B2", ... の形式で自動生成
        if brand_name is None:
            self.brand_name = [f"B{i+1}" for i in range(self.nbrand)]
        else:
            self.brand_name = brand_name

        # 基準期間(Period=1) での平均購入回数 M0 = カテゴリペネトレーション * 平均購入回数
        self.M0 = self.cat_pen * self.cat_buyrate
        # 現在の M は、分析期間を変更する際に更新されるが、初期値は M0
        self.M = self.M0

        # 1) Kパラメータ推定
        self.K = self._estimate_K(cat_pur_var)
        # 2) Sパラメータ推定
        self.S = self._estimate_S()

        # nstar が十分に大きいかチェック（分布の合計が1に近いか、平均が妥当か）
        prob_sum = sum(self._Pn(i) for i in range(self.nstar+1))
        mean_cat = sum(i * self._Pn(i) for i in range(self.nstar+1))
        if prob_sum < 0.99 or abs(mean_cat - self.M0) > 0.1:
            # 0.99 未満になるようなら、nstarが小さすぎて分布が取りきれていないかもしれない
            if self.check:
                print(f"[Warning] nstar={nstar} が分布を十分にカバーしていない可能性があります。")
            self.error = 1
        else:
            self.error = 0

        self.class_name = "dirichlet"

    def _estimate_K(self, cat_pur_var=None):
        """
        Kパラメータを推定する内部メソッド。
        
        cat_pur_varが None の場合:
          -> カテゴリペネトレーション（ゼロ購買）や平均購入回数をもとに最適化してKを求める
        cat_pur_varが指定されている場合:
          -> K = M^2 / (cat_pur_var - M) で直接計算（理論的な式に基づく）
        """
        if cat_pur_var is None:
            # cat_pen から導出される式を用いて K を最適化
            cp = math.log(1 - self.cat_pen)
            def eq1(k):
                # (k * log(1 + M/k) + log(1 - cat_pen))^2 を最小化する
                return (k * math.log(1 + self.M / k) + cp) ** 2
            # scipy.optimize の minimize_scalar で最適化
            res = minimize_scalar(eq1, bounds=(0.0001, self.max_K), method='bounded')
            return res.x
        else:
            # 分散を用いる式
            return (self.M**2) / (cat_pur_var - self.M)

    def _estimate_S(self):
        """
        Sパラメータを推定する内部メソッド。
        
        1) 各ブランドごとに最適なSを個別に探し
        2) 外れ値を削除
        3) シェア加重平均を取って最終的なSとする
        """
        if self.check:
            print("[Info] 各ブランドのSを推定中...")

        def find_opt_S_for_brand(j):
            # ブランド j に対して、理論ペネトレーションと観測値の二乗誤差が最小になるSを探す
            res = minimize_scalar(self._eq2, bounds=(0, self.max_S), method='bounded', args=(j,))
            if self.check:
                print(f"[Debug] Brand {j+1}, S={res.x:.4f}, Objective={res.fun:.6f}")
            return res.x

        # ブランド毎に S を計算
        S_list = [find_opt_S_for_brand(j) for j in range(self.nbrand)]
        bp = self._boxplot_stats(S_list)
        outlier = bp["out"]    # 箱ひげ図上の外れ値
        outlier2 = [s for s in S_list if s > bp["conf"][1]]  # ノッチ上限を越えるもの
        outliers = outlier + outlier2

        # 外れ値ではないものだけ残す
        schoose = [(s not in outliers) for s in S_list]
        if self.check and len(outliers) > 0:
            idx_out_brands = [i+1 for i, sc in enumerate(schoose) if not sc]
            print(f"[Info] 外れ値を除外: {outliers}, 対象ブランド: {idx_out_brands}")

        chosen_S = [S_list[i] for i, sc in enumerate(schoose) if sc]
        chosen_w = [self.brand_share[i] for i, sc in enumerate(schoose) if sc]

        # 全部外れ値になってしまった場合は fallback で平均値を使用
        if sum(chosen_w) == 0:
            S_final = np.mean(S_list)
        else:
            # シェア(brand_share)で加重平均
            S_final = np.average(chosen_S, weights=chosen_w)

        return S_final

    def _boxplot_stats(self, data):
        """
        箱ひげ図で用いられる外れ値検出・ノッチ範囲を計算する内部メソッド。
        
        返す値は以下のような辞書:
        {
          "out": 外れ値のリスト,
          "conf": [ノッチの下限, ノッチの上限]
        }
        """
        arr = np.array(data)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        median = np.median(arr)
        n = len(arr)

        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr
        outliers = arr[(arr < lower_whisker) | (arr > upper_whisker)]
        conf_lower = median - 1.58 * iqr / math.sqrt(n)
        conf_upper = median + 1.58 * iqr / math.sqrt(n)

        return {
            "out": outliers.tolist(),
            "conf": [conf_lower, conf_upper]
        }

    def _eq2(self, S, j):
        """
        理論ペネトレーション(カテゴリ*NBD×Dirichlet) と
        観測ペネトレーション(brand_pen_obs) の二乗誤差を返す。
        """
        t_pen = 1.0 - sum(self._Pn(i) * self._pzeron(i, j, S) for i in range(self.nstar+1))
        o_pen = self.brand_pen_obs[j]
        return (t_pen - o_pen) ** 2

    def _pzeron(self, n, j, S):
        """
        n回カテゴリを買うバイヤーが、ブランドjを0回も買わない確率。
        """
        alpha_j = S * self.brand_share[j]
        if n == 0:
            return 1.0
        return math.exp(sum(math.log(S - alpha_j + i) - math.log(S + i) for i in range(n)))

    def _p_rj_n(self, rj, n, j, S):
        """
        n回カテゴリを買うバイヤーが、ブランドjをrj回購入する確率。
        """
        alpha_j = S * self.brand_share[j]
        return comb(n, rj) * beta(alpha_j + rj, (S - alpha_j) + (n - rj)) / beta(alpha_j, (S - alpha_j))

    def _Pn(self, n):
        """
        NBD分布による、カテゴリをn回購入する確率。
        
        K, M に基づき:
        p(n) = 定数 * (K+...)/(1+...) の積の形で表される。
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
        解析対象期間を t倍に変更し、それに合わせて平均購買回数 M を更新する。
        （時間が2倍になれば M も2倍になる想定）
        """
        self.M = self.M0 * t

    def period_print(self):
        """
        現在の期間が基準期間(M0)の何倍になっているかを表示。
        """
        ratio = self.M / self.M0
        print(f"期間倍率: {round(ratio,2)}, 現在のM = {self.M}")

    def brand_pen(self, j):
        """
        現在パラメータ(M, S)下でのブランドjの理論ペネトレーションを計算。
        """
        p0 = sum(self._Pn(i) * self._p_rj_n(0, i, j, self.S) for i in range(self.nstar+1))
        return 1.0 - p0

    def brand_buyrate(self, j):
        """
        ブランドjバイヤー1人あたりの理論的購入回数を計算。
        """
        denom = self.brand_pen(j)
        if denom == 0:
            return 0.0

        def buyrate_n(n):
            return sum(r * self._p_rj_n(r, n, j, self.S) for r in range(1, n+1))

        numerator = sum(self._Pn(n) * buyrate_n(n) for n in range(1, self.nstar+1))
        return numerator / denom

    def wp(self, j):
        """
        ブランドj購入者のカテゴリ平均購入回数を計算。
        ( = n回中、ブランドjを買っている人が、そのn回全体で何回買っているかの平均)
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
        ブランド別に:
          - 実測ペネトレーション(Pen_Observed)
          - 理論ペネトレーション(Pen_Theoretical)
          - その差分(Pen_Diff)
          - 理論購入回数(BuyRate_Theoretical)
          - カテゴリ平均購入回数(WP_Theoretical)
        をまとめて DataFrame で返す。
        """
        records = []
        for j in range(self.nbrand):
            obs_pen = self.brand_pen_obs[j]  # 観測されたブランドペネトレーション
            thr_pen = self.brand_pen(j)      # 理論ペネトレーション
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
        推定された S, K や、基準期間の M0、現在の M, cat_pen, cat_buyrate などをまとめた DataFrameを返す。
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
    DirichletModel から集計レポートを作成するクラス。
    
    Rの summary.dirichlet が提供する "buy", "freq", "heavy", "dup" のようなレポートを、
    Python で再現する責務を担う。
    """

    def __init__(self, model):
        """
        DirichletModel のインスタンスを受け取り、サマリー出力に利用する。
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
        Rのsummary.dirichletに相当する集計を実行し、結果を辞書で返す。
        
        Parameters
        ----------
        t : float
            期間の倍率(例: 2.0なら2倍の期間)。model.period_set(t) を呼び出してから集計する。
        type : tuple(str)
            "buy","freq","heavy","dup" の中から出力したい集計を指定。
        digits : int
            小数点以下を何桁で四捨五入するか。
        freq_cutoff : int
            購入回数分布（freqサマリー）で、ここを超えると "freq_cutoff+1+" にまとめる。
        heavy_limit : iterable
            heavyサマリーで「ヘビーバイヤー」とみなすカテゴリ購入回数の範囲。（デフォルト 1..6）
        dup_brand : int
            dupサマリーで重複購買(duplication)を分析するブランドを1-basedインデックスで指定。
        
        Returns
        -------
        result : dict
            キーが "buy","freq","heavy","dup" のいずれかになり、
            値がサマリー結果（DataFrame or Series）となる辞書を返す。
        """
        # 期間をt倍に変更
        self.model.period_set(t)

        result = {}
        if heavy_limit is None:
            heavy_limit = range(1,7)

        # ローカル関数: heavyサマリーなどで部分的に使用
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
                # "buy"サマリー: ブランドごとの理論ペネトレーション(pen.brand), 理論購入回数(pur.brand), カテゴリ平均購入(pur.cat)を返す
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
                # "freq"サマリー: 0回,1回,...,freq_cutoff回, freq_cutoff+1回以上 の購入回数分布をブランド別に返す
                def prob_r(r, j):
                    s = 0.0
                    for n in range(r, self.model.nstar+1):
                        s += self.model._Pn(n) * self.model._p_rj_n(r, n, j, self.model.S)
                    return s

                arr = np.zeros((nbrand, freq_cutoff+2))
                for j in brand_idxs:
                    # 0..freq_cutoff 回
                    for r_ in range(freq_cutoff+1):
                        arr[j, r_] = prob_r(r_, j)
                    # freq_cutoff+1 回以上
                    tail_sum = 0.0
                    for n in range(freq_cutoff+1, self.model.nstar+1):
                        tail_sum += prob_r(n, j)
                    arr[j, freq_cutoff+1] = tail_sum

                col_names = list(range(freq_cutoff+1)) + [f"{freq_cutoff+1}+"]
                df_freq = pd.DataFrame(arr, index=self.model.brand_name, columns=col_names).round(digits)
                result["freq"] = df_freq

            elif tt == "heavy":
                # "heavy"サマリー: heavy_limitで指定したカテゴリ購入回数のバイヤーに限定したときの
                # ブランドペネトレーション & 購入頻度を計算
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
                        # ブランドjのバイヤーがheavyセグメント内で何回買っているか？
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
                # "dup"サマリー: duplication(重複購買)解析
                # dup_brand で指定したブランドを focal brand として、
                # そのブランド購買者が他のブランドをどの程度併買しているかを計算
                k_idx = dup_brand - 1
                r_dup = np.zeros(nbrand)
                r_dup[k_idx] = 1.0
                b_k = self.model.brand_pen(k_idx)
                other_idx = [x for x in brand_idxs if x != k_idx]

                for j in other_idx:
                    p0 = 0.0
                    for n in range(self.model.nstar+1):
                        alpha_sum = self.model.S * (self.model.brand_share[k_idx] + self.model.brand_share[j])
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

            else:
                # 不明なタイプはスキップ
                continue

        return result


# 実行例 (このファイルを単独で実行した場合にのみ動作)
if __name__ == "__main__":
    cat_pen_example = 0.6
    cat_buyrate_example = 2.0
    brand_share_example = [0.3, 0.2, 0.5]
    brand_pen_obs_example = [0.15, 0.10, 0.35]
    brand_name_example = ["BrandA", "BrandB", "BrandC"]

    # DirichletModelを作成
    model = DirichletModel(
        cat_pen=cat_pen_example,
        cat_buyrate=cat_buyrate_example,
        brand_share=brand_share_example,
        brand_pen_obs=brand_pen_obs_example,
        brand_name=brand_name_example,
        check=True
    )

    # サマリー用クラスを作成し、集計を行う
    reporter = DirichletSummary(model)

    summaries = reporter.summary_dirichlet(
        t=1,
        type=("buy","freq","heavy","dup"),
        digits=2,
        freq_cutoff=5,
        heavy_limit=range(1,7),
        dup_brand=1
    )

    print("=== Summary: buy ===")
    print(summaries["buy"], "\n")

    print("=== Summary: freq ===")
    print(summaries["freq"], "\n")

    print("=== Summary: heavy ===")
    print(summaries["heavy"], "\n")

    print("=== Summary: dup ===")
    print(summaries["dup"], "\n")

    # ブランド別メトリクス
    df_metrics = model.get_brand_metrics()
    print("=== get_brand_metrics() from DirichletModel ===")
    print(df_metrics, "\n")

    # 推定パラメータの一覧
    df_params = model.get_parameters_summary()
    print("=== get_parameters_summary() ===")
    print(df_params, "\n")
