"""
Test script to verify NaN handling fix in DecisionTreeEvaluator and XGBoostEvaluator
"""
import sys
sys.path.append('.')

import pandas as pd
from nucml_next.baselines import DecisionTreeEvaluator, XGBoostEvaluator

print("="*80)
print("TEST: NaN Handling Fix")
print("="*80)

# Create sample data with NaN values in AME features
# This simulates incomplete AME coverage
data = {
    'Z': [17, 17, 17, 92, 92, 92] * 100,
    'A': [35, 35, 35, 235, 235, 235] * 100,
    'N': [18, 18, 18, 143, 143, 143] * 100,
    'Energy': [1e5, 2e5, 3e5, 1e6, 2e6, 3e6] * 100,
    'CrossSection': [0.001, 0.002, 0.003, 1.5, 2.0, 2.5] * 100,
    'out_n': [0, 0, 0, 0, 0, 0] * 100,
    'out_p': [1, 1, 1, 0, 0, 0] * 100,
    'out_a': [0, 0, 0, 0, 0, 0] * 100,
    'out_g': [0, 0, 0, 0, 0, 0] * 100,
    'out_f': [0, 0, 0, 1, 1, 1] * 100,
    'out_t': [0, 0, 0, 0, 0, 0] * 100,
    'out_h': [0, 0, 0, 0, 0, 0] * 100,
    'out_d': [0, 0, 0, 0, 0, 0] * 100,
    'is_met': [0, 0, 0, 0, 0, 0] * 100,
    # AME features - simulate 50% coverage (half NaN)
    'Binding_Per_Nucleon_MeV': [8.5, 8.5, 8.5, float('nan'), float('nan'), float('nan')] * 100,
    'S_2n_MeV': [10.0, 10.0, 10.0, float('nan'), float('nan'), float('nan')] * 100,
    'Uncertainty': [0.01, 0.01, 0.01, 0.02, 0.02, 0.02] * 100,
    'Entry': ['ENTRY1'] * 600,
    'MT': [103, 103, 103, 18, 18, 18] * 100,
}

df = pd.DataFrame(data)

print(f"\nTest DataFrame:")
print(f"  Total rows: {len(df):,}")
print(f"  Rows with NaN in Binding_Per_Nucleon_MeV: {df['Binding_Per_Nucleon_MeV'].isna().sum():,}")
print(f"  Rows with NaN in S_2n_MeV: {df['S_2n_MeV'].isna().sum():,}")
print(f"  Expected rows after dropping NaN: {(~df['Binding_Per_Nucleon_MeV'].isna()).sum():,}")
print()

# Test 1: DecisionTreeEvaluator
print("="*80)
print("TEST 1: DecisionTreeEvaluator")
print("="*80)

try:
    dt_model = DecisionTreeEvaluator(max_depth=5, min_samples_leaf=2)

    print("\nTraining DecisionTreeEvaluator with NaN in features...")
    metrics = dt_model.train(df, test_size=0.2)

    print("\n✓ SUCCESS: DecisionTreeEvaluator handled NaN correctly!")
    print(f"  Test MSE: {metrics['test_mse']:.4e}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")

except Exception as e:
    print(f"\n✗ FAILED: {e}")

# Test 2: XGBoostEvaluator
print("\n" + "="*80)
print("TEST 2: XGBoostEvaluator")
print("="*80)

try:
    xgb_model = XGBoostEvaluator(max_depth=5, n_estimators=10)

    print("\nTraining XGBoostEvaluator with NaN in features...")
    metrics = xgb_model.train(df, test_size=0.2)

    print("\n✓ SUCCESS: XGBoostEvaluator handled NaN correctly!")
    print(f"  Test MSE: {metrics['test_mse']:.4e}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")

except Exception as e:
    print(f"\n✗ FAILED: {e}")

# Test 3: Test with 100% NaN (should fail with helpful error)
print("\n" + "="*80)
print("TEST 3: All NaN (Should Fail with Helpful Error)")
print("="*80)

df_all_nan = df.copy()
df_all_nan['Binding_Per_Nucleon_MeV'] = float('nan')
df_all_nan['S_2n_MeV'] = float('nan')

try:
    dt_model2 = DecisionTreeEvaluator(max_depth=5)
    print("\nAttempting to train with 100% NaN...")
    metrics = dt_model2.train(df_all_nan, test_size=0.2)
    print("\n✗ UNEXPECTED: Should have raised an error!")

except ValueError as e:
    print(f"\n✓ SUCCESS: Raised helpful error as expected!")
    print(f"  Error message preview: {str(e)[:100]}...")

except Exception as e:
    print(f"\n✗ FAILED with unexpected error: {e}")

print("\n" + "="*80)
print("ALL TESTS COMPLETE")
print("="*80)
