import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from collections import Counter

# 日本語表示のための設定
plt.rcParams['font.family'] = 'IPAexGothic'

# -------------------- 現実的なサンプルデータの生成 --------------------
def generate_realistic_sample_data(n_respondents=200, price_points=None):
    if price_points is None:
        price_points = [980, 1200, 1500, 1800, 2000, 2300, 2500]
    
    # 価格ポイントを昇順にソート
    price_points = sorted(price_points)
    
    data = []
    
    for respondent_id in range(n_respondents):
        # 各回答者のプライスポイント（閾値）をランダムに設定
        price_threshold = np.random.normal(1700, 400)
        
        # 各回答者に対してランダムな順序で価格を提示
        shuffled_prices = price_points.copy()
        np.random.shuffle(shuffled_prices)
        
        # 提示済みの価格と、最高受容価格を追跡
        presented_prices = []
        rejected_price_found = False
        max_acceptable_price = None
        
        # 各価格での回答を生成
        for price in shuffled_prices:
            # すでに拒否価格が見つかっており、現在の価格がそれより高い場合は質問しない
            if rejected_price_found and price > max_acceptable_price:
                continue
                
            # 価格が低いほど購入確率が高くなるように設定
            purchase_probability = 1 / (1 + np.exp((price - price_threshold) / 300))
            
            # 5段階評価に変換 (1=絶対に購入しない, 5=必ず購入する)
            intention = min(5, max(1, round(purchase_probability * 5)))
            
            # 回答を記録
            data.append({
                'respondent_id': respondent_id,
                'price': price,
                'purchase_intention': intention,
                'would_buy': 1 if intention >= 4 else 0  # 4以上なら購入する
            })
            
            presented_prices.append(price)
            
            # 購入意向が低い場合（3以下）、この価格を拒否として記録
            if intention <= 3:
                rejected_price_found = True
                max_acceptable_price = price
    
    # データフレームに変換
    df = pd.DataFrame(data)
    
    # データ収集の概要を表示
    print(f"総回答者数: {n_respondents}")
    print(f"総回答数: {len(df)}")
    print(f"理論上の全回答数（すべての価格を全員に提示した場合）: {n_respondents * len(price_points)}")
    print(f"データ取得率: {len(df) / (n_respondents * len(price_points)):.2%}")
    
    # 各価格ポイントでの回答数
    price_counts = Counter(df['price'])
    print("\n各価格ポイントでの回答数:")
    for price in sorted(price_counts.keys()):
        print(f"¥{price}: {price_counts[price]}件 ({price_counts[price]/n_respondents:.1%})")
    
    return df

# -------------------- 欠損データの補完 --------------------
def complete_missing_data(df, price_points):
    """
    打ち切られた調査データを補完する関数
    
    各回答者について、最後に回答した価格より高い価格では「購入しない」と見なす
    """
    total_respondents = df['respondent_id'].nunique()
    all_respondents = set(range(total_respondents))  # 全回答者のID
    
    # 補完後のデータを格納するリスト
    completed_data = []
    
    # 元のデータをまずコピー
    completed_data.extend(df.to_dict('records'))
    
    # 各価格ポイントについて処理
    for price in sorted(price_points):
        # その価格に回答した回答者のIDを取得
        respondents_for_price = set(df[df['price'] == price]['respondent_id'])
        
        # 回答していない回答者を特定
        missing_respondents = all_respondents - respondents_for_price
        
        # 回答していない理由を判定
        for resp_id in missing_respondents:
            # 回答者のデータを取得
            resp_data = df[df['respondent_id'] == resp_id]
            
            # 提示された最高価格を取得
            highest_price_presented = resp_data['price'].max() if not resp_data.empty else 0
            
            # 補完が必要な条件を論理的に考慮
            # 1. その価格が提示されていない
            if price not in resp_data['price'].values:
                # この回答者の回答した価格と購入意向を取得
                resp_prices = resp_data['price'].values
                resp_would_buy = resp_data['would_buy'].values
                
                # ケース1: 提示された価格の中に、現在の価格より低い価格で「購入しない」があるか
                lower_prices = resp_prices[resp_prices < price]
                if len(lower_prices) > 0:
                    lower_responses = resp_data[resp_data['price'].isin(lower_prices)]
                    if any(lower_responses['would_buy'] == 0):
                        # より低い価格で拒否している -> 「購入しない」と補完
                        completed_data.append({
                            'respondent_id': resp_id,
                            'price': price,
                            'purchase_intention': 1,  # 最低評価
                            'would_buy': 0,  # 購入しない
                            'is_imputed': 1  # 補完データであることを示すフラグ
                        })
                        continue
                
                # ケース2: 提示された価格の中に、現在の価格より高い価格で「購入する」があるか
                higher_prices = resp_prices[resp_prices > price]
                if len(higher_prices) > 0:
                    higher_responses = resp_data[resp_data['price'].isin(higher_prices)]
                    if any(higher_responses['would_buy'] == 1):
                        # より高い価格で購入している -> 「購入する」と補完
                        completed_data.append({
                            'respondent_id': resp_id,
                            'price': price,
                            'purchase_intention': 4,  # 高評価（購入する）
                            'would_buy': 1,  # 購入する
                            'is_imputed': 1  # 補完データであることを示すフラグ
                        })
                        continue
                
                # ケース3: 提示された最高価格より高い価格は補完しない（不明なため）
                # ケース4: 提示された最低価格より低い価格は「購入する」と補完
                if len(resp_prices) > 0:
                    min_price = min(resp_prices)
                    min_response = resp_data[resp_data['price'] == min_price]['would_buy'].iloc[0]
                    
                    if price < min_price and min_response == 1:
                        # 最低提示価格でも購入している -> より低い価格でも「購入する」と補完
                        completed_data.append({
                            'respondent_id': resp_id,
                            'price': price,
                            'purchase_intention': 4,  # 高評価（購入する）
                            'would_buy': 1,  # 購入する
                            'is_imputed': 1  # 補完データであることを示すフラグ
                        })
    
    # 元のデータに補完フラグを追加
    for item in completed_data[:len(df)]:
        item['is_imputed'] = 0
    
    # 補完後のデータをデータフレームに変換
    completed_df = pd.DataFrame(completed_data)
    
    # 補完データの統計を表示
    n_original = len(df)
    n_completed = len(completed_df)
    n_imputed = n_completed - n_original
    
    print(f"\nデータ補完の統計:")
    print(f"元のデータ数: {n_original}")
    print(f"補完後のデータ数: {n_completed}")
    print(f"補完されたデータ数: {n_imputed} ({n_imputed/n_completed:.1%})")
    
    return completed_df

# -------------------- 基本分析 --------------------
def basic_analysis(df, price_points, total_respondents=None):
    # 全回答者数を取得
    if total_respondents is None:
        total_respondents = df['respondent_id'].nunique()
    
    # 各価格での購入意向集計
    results = []
    for price in price_points:
        price_data = df[df['price'] == price]
        
        # 回答数を確認
        n_responses = len(price_data)
        n_original = sum(price_data['is_imputed'] == 0) if 'is_imputed' in price_data.columns else n_responses
        
        if n_responses > 0:
            # 購入率を複数の方法で計算
            
            # 1. 回答者のみでの購入率（従来の方法）
            raw_purchase_rate = (price_data['would_buy'] == 1).mean()
            
            # 2. 全回答者に対する購入率（補完あり）
            n_would_buy = sum(price_data['would_buy'] == 1)
            purchase_rate = n_would_buy / total_respondents
            
            # 期待収益計算
            expected_revenue = price * purchase_rate
            
            results.append({
                'price': price,
                'n_responses': n_responses,
                'n_original': n_original,
                'n_imputed': n_responses - n_original if 'is_imputed' in price_data.columns else 0,
                'response_rate': n_responses / total_respondents,
                'raw_purchase_rate': raw_purchase_rate,  # 回答者のみでの購入率
                'purchase_rate': purchase_rate,  # 全回答者に対する購入率
                'expected_revenue': expected_revenue
            })
    
    results_df = pd.DataFrame(results)
    
    # 収益最大化価格の特定（データがある場合のみ）
    if not results_df.empty:
        max_revenue_idx = results_df['expected_revenue'].idxmax()
        optimal_price = results_df.loc[max_revenue_idx, 'price']
        max_revenue = results_df.loc[max_revenue_idx, 'expected_revenue']
        
        print(f"\n収益最大化価格: {optimal_price}円")
        print(f"最大期待収益: {max_revenue:.2f}")
        print(f"この価格の回答数: {results_df.loc[max_revenue_idx, 'n_responses']}件")
        print(f"回答率: {results_df.loc[max_revenue_idx, 'response_rate']:.1%}")
        print(f"回答者のみの購入率: {results_df.loc[max_revenue_idx, 'raw_purchase_rate']:.1%}")
        print(f"全回答者に対する購入率: {results_df.loc[max_revenue_idx, 'purchase_rate']:.1%}")
    
    return results_df

# -------------------- 回帰分析 --------------------
def regression_analysis(df, price_points, total_respondents=None):
    # 全回答者ベースの購入率を算出
    if total_respondents is None:
        total_respondents = df['respondent_id'].nunique()
    
    purchase_rates = []
    for price in sorted(price_points):
        price_data = df[df['price'] == price]
        n_would_buy = sum(price_data['would_buy'] == 1)
        purchase_rate = n_would_buy / total_respondents
        n_responses = len(price_data)
        
        purchase_rates.append({
            'price': price,
            'purchase_rate': purchase_rate,
            'count': n_responses
        })
    
    # データフレームに変換
    agg_data = pd.DataFrame(purchase_rates)
    
    # 回答数を重みとして使用
    weights = agg_data['count'] / agg_data['count'].sum()
    
    # 線形回帰
    X = agg_data['price']
    X = sm.add_constant(X)
    y = agg_data['purchase_rate']
    
    model = sm.WLS(y, X, weights=weights).fit()
    print("\n線形回帰モデル（重み付き）:")
    print(model.summary())
    
    # 2次回帰（曲線）モデル
    X_quad = agg_data[['price']]
    X_quad['price_squared'] = X_quad['price'] ** 2
    X_quad = sm.add_constant(X_quad)
    
    quad_model = sm.WLS(y, X_quad, weights=weights).fit()
    print("\n2次回帰モデル（重み付き）:")
    print(quad_model.summary())
    
    # 予測値計算
    price_range = np.linspace(min(price_points), max(price_points), 100)
    X_pred = sm.add_constant(np.column_stack((price_range, price_range**2)))
    y_pred = quad_model.predict(X_pred)
    
    # 予測購入率から収益を計算
    predicted_revenue = price_range * y_pred
    
    # 理論上の最適価格を計算
    optimal_idx = np.argmax(predicted_revenue)
    theoretical_optimal_price = price_range[optimal_idx]
    max_predicted_revenue = predicted_revenue[optimal_idx]
    
    print(f"\n理論上の収益最大化価格: {theoretical_optimal_price:.2f}円")
    print(f"理論上の最大期待収益: {max_predicted_revenue:.2f}")
    
    return price_range, y_pred, predicted_revenue, agg_data

# -------------------- ロジスティック回帰 --------------------
def logistic_regression_analysis(df):
    # 補完データを含む場合は補完フラグも特徴量として使用
    if 'is_imputed' in df.columns:
        # 補完データの影響を適切に処理
        # 実データのみを使用してモデルを構築
        real_data = df[df['is_imputed'] == 0]
        
        X = real_data[['price']]
        y = real_data['would_buy']
        
        # 回答者ごとの回答数をカウント
        resp_counts = real_data['respondent_id'].value_counts()
        
        # カウントの逆数をウェイトとして使用（少ないデータの回答者の意見を重視）
        sample_weights = real_data['respondent_id'].map(lambda x: 1/resp_counts[x])
        
        model = LogisticRegression(random_state=42)
        model.fit(X, y, sample_weight=sample_weights)
        
        print("\nロジスティック回帰モデル（実データのみ、ウェイト調整済み）:")
    else:
        # 通常のデータ処理
        X = df[['price']]
        y = df['would_buy']
        
        # 回答者ごとの回答数をカウント
        resp_counts = df['respondent_id'].value_counts()
        
        # カウントの逆数をウェイトとして使用
        sample_weights = df['respondent_id'].map(lambda x: 1/resp_counts[x])
        
        model = LogisticRegression(random_state=42)
        model.fit(X, y, sample_weight=sample_weights)
        
        print("\nロジスティック回帰モデル（ウェイト調整済み）:")
    
    print(f"係数: {model.coef_[0][0]:.6f}")
    print(f"切片: {model.intercept_[0]:.6f}")
    
    # 価格感度計算
    price_sensitivity = abs(model.coef_[0][0])
    print(f"価格感度: {price_sensitivity:.6f}")
    
    # 予測
    price_range = np.linspace(min(df['price']), max(df['price']), 100).reshape(-1, 1)
    purchase_prob = model.predict_proba(price_range)[:, 1]
    
    # 予測収益
    predicted_revenue_logit = price_range.flatten() * purchase_prob
    
    # 最適価格
    optimal_idx = np.argmax(predicted_revenue_logit)
    optimal_price_logit = price_range[optimal_idx][0]
    
    print(f"ロジスティックモデルでの最適価格: {optimal_price_logit:.2f}円")
    
    return price_range.flatten(), purchase_prob, predicted_revenue_logit

# -------------------- 価格弾力性計算 --------------------
def calculate_price_elasticity(df, price_points, total_respondents=None):
    # 全回答者ベースの購入率を使用
    if total_respondents is None:
        total_respondents = df['respondent_id'].nunique()
    
    # 価格弾力性: (需要の変化率)/(価格の変化率)
    elasticities = []
    
    # 各価格ポイントでの購入率を計算
    purchase_rates = []
    sample_counts = []
    
    for price in sorted(price_points):
        data = df[df['price'] == price]
        if len(data) > 0:  # データがある場合のみ
            # 全回答者ベースの購入率
            n_would_buy = sum(data['would_buy'] == 1)
            rate = n_would_buy / total_respondents
            
            purchase_rates.append(rate)
            sample_counts.append(len(data))
        else:
            purchase_rates.append(None)
            sample_counts.append(0)
    
    # 価格弾力性を計算（データがある連続した価格ポイント間のみ）
    for i in range(1, len(price_points)):
        if purchase_rates[i-1] is not None and purchase_rates[i] is not None:
            p1, p2 = price_points[i-1], price_points[i]
            q1, q2 = purchase_rates[i-1], purchase_rates[i]
            n1, n2 = sample_counts[i-1], sample_counts[i]
            
            # 弧弾力性の計算
            # ε = ((q2-q1)/((q2+q1)/2)) / ((p2-p1)/((p2+p1)/2))
            price_change_pct = (p2 - p1) / ((p2 + p1) / 2)
            demand_change_pct = (q2 - q1) / ((q2 + q1) / 2)
            
            if price_change_pct != 0:
                elasticity = demand_change_pct / price_change_pct
            else:
                elasticity = 0
            
            # サンプルサイズも記録
            elasticities.append({
                'price_range': f"{p1}円-{p2}円",
                'price_midpoint': (p1 + p2) / 2,
                'elasticity': elasticity,
                'n_samples': min(n1, n2)  # より少ない方をサンプルサイズとして使用
            })
    
    if elasticities:
        elasticity_df = pd.DataFrame(elasticities)
        print("\n価格弾力性:")
        print(elasticity_df)
        return elasticity_df
    else:
        print("\n価格弾力性を計算するのに十分なデータがありません")
        return pd.DataFrame()

# -------------------- 可視化 --------------------
def visualize_results(basic_results, price_range, predicted_prob, predicted_revenue, 
                      logit_price_range, logit_prob, logit_revenue, elasticity_df,
                      agg_data, df, price_points, total_respondents=None):
    
    if total_respondents is None:
        total_respondents = df['respondent_id'].nunique()
    
    # 1. 購入確率曲線
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    # 実際のデータ
    plt.scatter(agg_data['price'], agg_data['purchase_rate'], 
                s=agg_data['count']*0.5, alpha=0.7, label='実際のデータ')
    
    # 生データと補完データの比較（もし補完データがある場合）
    if 'is_imputed' in df.columns:
        # 原データのみでの購入率（各価格ポイントごと）
        raw_rates = []
        for price in sorted(price_points):
            raw_data = df[(df['price'] == price) & (df['is_imputed'] == 0)]
            if len(raw_data) > 0:
                raw_rate = raw_data['would_buy'].mean()
                raw_rates.append({'price': price, 'raw_rate': raw_rate})
        
        if raw_rates:
            raw_df = pd.DataFrame(raw_rates)
            plt.plot(raw_df['price'], raw_df['raw_rate'], 'o--', color='orange', 
                    alpha=0.5, label='補完前購入率')
    
    # モデル予測
    plt.plot(price_range, predicted_prob, '--', label='2次回帰予測')
    plt.plot(logit_price_range, logit_prob, ':', label='ロジスティック回帰予測')
    
    plt.xlabel('価格 (円)')
    plt.ylabel('購入確率')
    plt.title('価格に対する購入確率（円の大きさはサンプル数）')
    plt.grid(True)
    plt.legend()
    
    # 2. 収益曲線
    plt.subplot(2, 2, 2)
    plt.scatter(basic_results['price'], basic_results['expected_revenue'], 
                s=basic_results['n_responses']*0.5, alpha=0.7, label='実際のデータ')
    plt.plot(price_range, predicted_revenue, '--', label='2次回帰予測')
    plt.plot(logit_price_range, logit_revenue, ':', label='ロジスティック回帰予測')
    
    # 購入率の比較グラフを小さく追加
    ax2 = plt.gca().twinx()
    ax2.plot(basic_results['price'], basic_results['raw_purchase_rate'], 'o--', color='green', alpha=0.5, label='回答者のみの購入率')
    ax2.plot(basic_results['price'], basic_results['purchase_rate'], 'o-', color='red', alpha=0.5, label='全体購入率')
    ax2.set_ylabel('購入率')
    ax2.legend(loc='upper right')
    
    # 最適価格を表示
    if not basic_results.empty:
        optimal_price = basic_results.loc[basic_results['expected_revenue'].idxmax(), 'price']
        plt.axvline(x=optimal_price, color='r', linestyle='-', alpha=0.3, label=f'最適価格: {optimal_price}円')
    
    plt.xlabel('価格 (円)')
    plt.ylabel('期待収益')
    plt.title('価格に対する期待収益（円の大きさはサンプル数）')
    plt.grid(True)
    plt.legend()
    
    # 3. 価格弾力性
    plt.subplot(2, 2, 3)
    if not elasticity_df.empty:
        # 弾力性の大きさとサンプルサイズを反映
        plt.bar(elasticity_df['price_range'], elasticity_df['elasticity'],
                alpha=0.7, width=0.6)
        
        # サンプルサイズのオーバーレイ
        for i, row in elasticity_df.iterrows():
            plt.text(i, row['elasticity'] - 0.1 if row['elasticity'] < 0 else row['elasticity'] + 0.1, 
                     f"n={row['n_samples']}", ha='center')
        
        plt.axhline(y=-1, color='r', linestyle='--', label='単位弾力性 (ε=-1)')
        plt.xlabel('価格帯')
        plt.ylabel('価格弾力性')
        plt.title('価格帯別の価格弾力性（テキストはサンプル数）')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.legend()
    else:
        plt.text(0.5, 0.5, '価格弾力性を計算するのに十分なデータがありません', 
                 ha='center', va='center', transform=plt.gca().transAxes)
    
    # 4. 補完データの影響
    plt.subplot(2, 2, 4)
    
    if 'is_imputed' in df.columns:
        # 各価格ポイントでの実データと補完データの比率
        imputed_stats = []
        for price in sorted(price_points):
            price_data = df[df['price'] == price]
            n_total = len(price_data)
            if n_total > 0:
                n_real = sum(price_data['is_imputed'] == 0)
                n_imputed = sum(price_data['is_imputed'] == 1)
                imputed_stats.append({
                    'price': price,
                    'n_real': n_real,
                    'n_imputed': n_imputed,
                    'imputed_ratio': n_imputed / n_total if n_total > 0 else 0
                })
        
        imputed_df = pd.DataFrame(imputed_stats)
        
        # 積み上げ棒グラフで表示
        width = (max(price_points) - min(price_points)) / (len(price_points) * 2.5)
        
        plt.bar(imputed_df['price'], imputed_df['n_real'], width=width, 
                label='実データ', alpha=0.7, color='blue')
        plt.bar(imputed_df['price'], imputed_df['n_imputed'], width=width, 
                bottom=imputed_df['n_real'], label='補完データ', alpha=0.7, color='red')
        
        # 比率を右軸に表示
        ax2 = plt.gca().twinx()
        ax2.plot(imputed_df['price'], imputed_df['imputed_ratio'], 'o--', color='green', label='補完データ比率')
        ax2.set_ylabel('補完データ比率')
        ax2.set_ylim(0, 1)
        
        # 各軸のレジェンド
        plt.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.xlabel('価格 (円)')
        plt.ylabel('回答数')
        plt.title('各価格での実データと補完データの比較')
    else:
        # 補完データがない場合は別の情報を表示
        response_counts = df['price'].value_counts().sort_index()
        
        response_rates = {}
        for price in price_points:
            if price in response_counts:
                response_rates[price] = response_counts[price] / total_respondents
            else:
                response_rates[price] = 0
        
        # 取得率のグラフ
        plt.bar(response_rates.keys(), response_rates.values())
        plt.xlabel('価格 (円)')
        plt.ylabel('回答取得率')
        plt.title('各価格ポイントでの回答取得率')
        plt.ylim(0, 1.0)
        
        # 取得率の数値を表示
        for price, rate in response_rates.items():
            plt.text(price, rate + 0.02, f'{rate:.0%}', ha='center')
    
    plt.tight_layout()
    plt.show()

# -------------------- メイン処理 --------------------
# 設定
np.random.seed(42)
price_points = [980, 1200, 1500, 1800, 2000, 2300, 2500]

# 現実的なサンプルデータ生成
df = generate_realistic_sample_data(n_respondents=200, price_points=price_points)
print("\nサンプルデータ（一部）:")
print(df.head())

# 欠損データの補完
completed_df = complete_missing_data(df, price_points)

# 補完前データでの基本分析
print("\n【補完前データでの分析】")
basic_results_raw = basic_analysis(df, price_points)

# 補完後データでの基本分析
print("\n【補完後データでの分析】")
total_respondents = df['respondent_id'].nunique()  # 全回答者数
basic_results = basic_analysis(completed_df, price_points, total_respondents)

# 補完後データでの回帰分析
price_range, predicted_prob, predicted_revenue, agg_data = regression_analysis(completed_df, price_points, total_respondents)

# 補完後データでのロジスティック回帰
logit_price_range, logit_prob, logit_revenue = logistic_regression_analysis(completed_df)

# 補完後データでの価格弾力性
elasticity_df = calculate_price_elasticity(completed_df, price_points, total_respondents)

# 結果の可視化
visualize_results(basic_results, price_range, predicted_prob, predicted_revenue, 
                  logit_price_range, logit_prob, logit_revenue, elasticity_df,
                  agg_data, completed_df, price_points, total_respondents)

# 追加の分析: 回答者ごとの最大受容価格の分布
max_acceptance_prices = []

for respondent_id in df['respondent_id'].unique():
    # 回答者のデータを取得
    respondent_data = df[df['respondent_id'] == respondent_id]
    
    # 購入すると回答した最大価格を特定
    if any(respondent_data['would_buy'] == 1):
        purchase_data = respondent_data[respondent_data['would_buy'] == 1]
        max_price = purchase_data['price'].max()
        max_acceptance_prices.append(max_price)

if max_acceptance_prices:
    plt.figure(figsize=(10, 6))
    plt.hist(max_acceptance_prices, bins=len(price_points), alpha=0.7)
    plt.axvline(x=np.median(max_acceptance_prices), color='r', linestyle='--', 
                label=f'中央値: {np.median(max_acceptance_prices)}円')
    plt.axvline(x=np.mean(max_acceptance_prices), color='g', linestyle='--', 
                label=f'平均値: {np.mean(max_acceptance_prices):.0f}円')
    
    plt.xlabel('最高受容価格 (円)')
    plt.ylabel('回答者数')
    plt.title('回答者ごとの最高受容価格の分布')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    print("\n最高受容価格の統計:")
    print(f"平均: {np.mean(max_acceptance_prices):.0f}円")
    print(f"中央値: {np.median(max_acceptance_prices)}円")
    print(f"最頻値: {pd.Series(max_acceptance_prices).mode()[0]}円")
  
