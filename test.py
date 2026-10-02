from scipy.stats import ttest_ind, ttest_rel


# ============================================================
# QUESTION 1: INDEPENDENT TWO-SAMPLE T-TEST
# ============================================================

group_A = [65, 70, 68, 72, 66, 75, 69, 71, 67, 70]
group_B = [72, 75, 78, 74, 80, 77, 73, 79, 76, 81]

alpha = 0.05

t_statistic, p_value = ttest_ind(group_A, group_B)

print("=" * 60)
print("QUESTION 1: INDEPENDENT TWO-SAMPLE T-TEST")
print("=" * 60)

print("Group A:", group_A)
print("Group B:", group_B)
print("Alpha:", alpha)
print("T-statistic:", t_statistic)
print("P-value:", p_value)

if p_value < alpha:
    print("Result: There is a statistically significant difference between the two teaching methods.")
else:
    print("Result: There is no statistically significant difference between the two teaching methods.")


# ============================================================
# QUESTION 2: PAIRED T-TEST
# ============================================================

before = [52, 48, 65, 60, 55, 70, 58, 62, 50, 67]
after = [60, 55, 70, 66, 63, 75, 64, 68, 57, 72]

t_statistic, p_value = ttest_rel(before, after)

print("\n")
print("=" * 60)
print("QUESTION 2: PAIRED T-TEST")
print("=" * 60)

print("Before:", before)
print("After :", after)
print("Alpha:", alpha)
print("T-statistic:", t_statistic)
print("P-value:", p_value)

if p_value < alpha:
    print("Result: The training program resulted in a statistically significant change in programming scores.")
else:
    print("Result: The training program did not result in a statistically significant change in programming scores.")


# ============================================================
# QUESTION 3: INDEPENDENT T-TEST VS PAIRED T-TEST
# ============================================================

independent_group_1 = [60, 62, 65, 67, 64, 68, 61, 66, 63, 69]
independent_group_2 = [70, 72, 75, 73, 78, 76, 71, 74, 77, 79]

paired_before = [55, 60, 58, 62, 65, 57, 61, 59, 63, 56]
paired_after = [60, 65, 63, 67, 70, 62, 66, 64, 68, 61]

ind_t_statistic, ind_p_value = ttest_ind(
    independent_group_1,
    independent_group_2
)

paired_t_statistic, paired_p_value = ttest_rel(
    paired_before,
    paired_after
)

print("\n")
print("=" * 60)
print("QUESTION 3: INDEPENDENT VS PAIRED T-TEST")
print("=" * 60)

print("\nIndependent T-Test")
print("-" * 30)
print("Group 1:", independent_group_1)
print("Group 2:", independent_group_2)
print("T-statistic:", ind_t_statistic)
print("P-value:", ind_p_value)

if ind_p_value < alpha:
    print("Result: Statistically significant difference.")
else:
    print("Result: No statistically significant difference.")

print("\nPaired T-Test")
print("-" * 30)
print("Before:", paired_before)
print("After :", paired_after)
print("T-statistic:", paired_t_statistic)
print("P-value:", paired_p_value)

if paired_p_value < alpha:
    print("Result: Statistically significant change.")
else:
    print("Result: No statistically significant change.")


print("\n")
print("=" * 60)
print("EXPLANATION")
print("=" * 60)

print("Independent t-test is used when the two groups contain")
print("different and unrelated observations.")

print("\nPaired t-test is used when the observations are related,")
print("such as measurements taken from the same students before")
print("and after training.")

print("\nTherefore, the tests can produce different t-statistics")
print("and p-values because they use different assumptions and")
print("different relationships between the observations.")