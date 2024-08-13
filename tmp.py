import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# シミュレーションのパラメータ
alpha = 0.05  # 有意水準
desired_power = 0.8  # 目標検出力
num_simulations = 10000  # シミュレーションの回数
sample_sizes = np.arange(500, 15000, 1000)  # サンプルサイズの範囲

def calculate_power(n, p_control, p_treatment, alpha, num_simulations):
    """特定のサンプルサイズでの検出力を計算する関数
    
    Args:
        n (int): サンプルサイズ
        p_control (float): コントロールグループのCTR
        p_treatment (float): トリートメントグループのCTR
        alpha (float): 有意水準
        num_simulations (int): シミュレーションの回数
    
    Returns:
        float: 検出力
    """
    # コントロールグループとトリートメントグループの成功回数をシミュレート
    successes_control = np.random.binomial(n, p_control, num_simulations)
    successes_treatment = np.random.binomial(n, p_treatment, num_simulations)
    
    # プールされた確率と標準誤差を計算
    pooled_prob = (successes_control + successes_treatment) / (2 * n)
    pooled_se = np.sqrt(2 * pooled_prob * (1 - pooled_prob) / n)
    
    # zスコアを計算
    z_scores = (successes_treatment / n - successes_control / n) / pooled_se
    
    # 臨界値を計算
    critical_value = stats.norm.ppf(1 - alpha / 2)
    
    # 検出力を計算
    power = np.mean(np.abs(z_scores) > critical_value)
    return power

def plot_power_analysis(p_control, p_treatment):
    """検出力分析をプロットする関数
    
    Args:
        p_control (float): コントロールグループのCTR
        p_treatment (float): トリートメントグループのCTR
    """
    results = []
    for n in sample_sizes:
        try:
            current_power = calculate_power(n, p_control, p_treatment, alpha, num_simulations)
            results.append((n, current_power))
            # print(f"Sample size: {n}, Power: {current_power:.4f}")
        except Exception as e:
            print(f"Error calculating power for sample size {n}: {e}")

    # 結果をプロット
    plt.figure(figsize=(10, 6))
    sample_sizes_vals, powers = zip(*results)
    plt.plot(sample_sizes_vals, powers, marker='o', linestyle='-', color='b', label='Power')
    plt.axhline(y=desired_power, color='r', linestyle='--', label='Desired Power (0.8)')
    plt.xlabel('Sample Size')
    plt.ylabel('Power')
    plt.title('Power Analysis for A/B Testing (CTR)')
    plt.legend()
    plt.grid(True)

    # 注釈を追加
    for (n, power) in results:
        plt.annotate(f'{power:.2f}', xy=(n, power), xytext=(5, 5), textcoords='offset points')

    plt.show()

# インタラクティブウィジェットを作成
p_control_slider = widgets.FloatSlider(value=0.05, min=0.01, max=0.10, step=0.001, readout_format='.003f', description='p_control:')
p_treatment_slider = widgets.FloatSlider(value=0.06, min=0.01, max=0.10, step=0.001, readout_format='.003f', description='p_treatment:')
ui = widgets.HBox([p_control_slider, p_treatment_slider])

out = widgets.interactive_output(plot_power_analysis, {'p_control': p_control_slider, 'p_treatment': p_treatment_slider})

display(ui, out)
