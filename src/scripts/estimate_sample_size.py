import statsmodels.stats.proportion as smp_prop
import statsmodels.stats.power as smp_power
import numpy as np

# --- User-Defined Parameters ---
p0_reference_accuracy = 0.9189

p1_expected_accuracy = p0_reference_accuracy # Assumed true accuracy for power calculation
                              # Often set equal to p0 for non-inferiority designs

alpha = 0.05               # Significance level (one-sided for NI)
power = 0.80               # Desired statistical power
# --- End of User-Defined Parameters ---

# Define the non-inferiority threshold
print(f"Reference Accuracy (p0): {p0_reference_accuracy:.4f}")
print(f"Alpha: {alpha}, Power: {power}")
for p_threshold in [0.75, 0.8, 0.85, 0.9]:
    # Calculate effect size (Cohen's h) between expected and NI threshold
    effect_size = smp_prop.proportion_effectsize(prop1=p1_expected_accuracy,
                                                    prop2=p_threshold)
    # Initialize power analysis object
    analysis = smp_power.NormalIndPower()
    # Calculate sample size
    # Alternative is 'larger' because H1 is p > (p0 - delta)
    sample_size = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=0, # Signalling one-sample interpretation
        alternative='larger',
        nobs1=None
    )
    print("")
    print(f"Non-Inferiority Threshold (p0 - delta): {p_threshold:.4f}")
    print(f"Non-Inferiority Margin (delta): {p0_reference_accuracy-p_threshold:.4f}")
    print(f"Assumed True Prospective Accuracy (p1): {p1_expected_accuracy:.4f}")
    print(f"Calculated Effect Size (Cohen's h): {effect_size:.4f}")
    print(f"Required sample size for Non-Inferiority: {np.ceil(sample_size):.0f}")

# Plot sample size for different power levels
power_levels = np.linspace(0.7, 0.95, 6)
sample_sizes = []

for p in power_levels:
    ss = smp.power_proportions_1samp(
        value=expected_accuracy,
        null_value=target_accuracy,
        alpha=alpha,
        power=p,
        alternative=alternative
    )
    sample_sizes.append(np.ceil(ss))

# Plot the effect of power on sample size
plt.figure(figsize=(10, 6))
plt.plot(power_levels, sample_sizes, 'bo-', linewidth=2)
plt.xlabel('Statistical Power')
plt.ylabel('Required Sample Size')
plt.title('Sample Size vs. Statistical Power')
plt.grid(True)
plt.xticks(power_levels)
plt.yticks(sample_sizes)

# Calculate sample sizes for different expected accuracies
expected_accuracies = np.linspace(0.94, 0.98, 9)
sample_sizes_by_accuracy = []

for acc in expected_accuracies:
    # Skip cases where expected accuracy exceeds target
    if acc >= target_accuracy:
        sample_sizes_by_accuracy.append(np.nan)
        continue
        
    ss = smp.power_proportions_1samp(
        value=acc,
        null_value=target_accuracy,
        alpha=alpha,
        power=power,
        alternative=alternative
    )
    sample_sizes_by_accuracy.append(np.ceil(ss))

# Plot the effect of expected accuracy on sample size
plt.figure(figsize=(10, 6))
plt.plot(expected_accuracies, sample_sizes_by_accuracy, 'ro-', linewidth=2)
plt.xlabel('Expected Accuracy')
plt.ylabel('Required Sample Size')
plt.title('Sample Size vs. Expected Accuracy (Target=97%)')
plt.grid(True)
plt.ylim(bottom=0)

# Function to calculate confidence interval width for a given sample size
def ci_width(n, p=expected_accuracy, alpha=0.05):
    """Calculate confidence interval width for a proportion."""
    # Wilson score interval
    from statsmodels.stats.proportion import proportion_confint
    lower, upper = proportion_confint(count=int(n*p), nobs=n, alpha=alpha, method='wilson')
    return upper - lower

# Calculate CI widths for different sample sizes
sample_sizes_ci = np.arange(100, 2100, 100)
ci_widths = [ci_width(n) for n in sample_sizes_ci]

# Plot sample size vs. confidence interval width
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes_ci, ci_widths, 'go-', linewidth=2)
plt.xlabel('Sample Size')
plt.ylabel('95% Confidence Interval Width')
plt.title('Confidence Interval Width vs. Sample Size')
plt.grid(True)
plt.axhline(y=0.01, color='r', linestyle='--', label='1% width')
plt.axhline(y=0.02, color='b', linestyle='--', label='2% width')
plt.legend()

# Account for class imbalance
# NOTE: Sag, Trans, Bladder
class_distribution = [0.5, 0.36, 0.14]  # Given distribution
min_class_proportion = min(class_distribution)
min_class_size_needed = 30  # Minimum samples per class for reliable statistics

total_adjusted_for_imbalance = min_class_size_needed / min_class_proportion

print(f"Considering class imbalance, minimum recommended sample size: {np.ceil(total_adjusted_for_imbalance)}")

# Calculate adjusted sample size considering both statistical power and class imbalance
final_recommended_sample_size = max(sample_size, total_adjusted_for_imbalance)
print(f"Final recommended sample size: {np.ceil(final_recommended_sample_size)}")