# ============================================================================
# DIAGNOSTIC: Why are predictions horizontal lines?
# ============================================================================
# Add this cell after training the Decision Tree model to diagnose the issue

print("="*80)
print("DIAGNOSTIC: Investigating Horizontal Line Predictions")
print("="*80)

# 1. Check training data Energy values
print("\n1. Training Data Energy Check:")
print(f"   Energy min: {df_naive['Energy'].min():.6e}")
print(f"   Energy max: {df_naive['Energy'].max():.6e}")
print(f"   Energy mean: {df_naive['Energy'].mean():.6e}")
print(f"   Energy median: {df_naive['Energy'].median():.6e}")
print(f"\n   First 10 Energy values:")
print(f"   {df_naive['Energy'].head(10).values}")

# 2. Check feature importance
print("\n2. Feature Importance:")
importance = dt_model.get_feature_importance()
print(importance.head(15))

energy_imp = importance[importance['Feature'] == 'Energy']
if len(energy_imp) > 0:
    energy_imp_value = energy_imp['Importance'].values[0]
    print(f"\n   ✓ Energy importance: {energy_imp_value:.6f}")
    if energy_imp_value < 0.01:
        print("   ⚠️  WARNING: Energy has very low importance (<0.01)")
        print("   This means the model is essentially ignoring Energy!")
    else:
        print(f"   ✓ Energy is being used ({energy_imp_value:.1%} importance)")
else:
    print("\n   ❌ ERROR: Energy not found in feature importance!")

# 3. Check what features the model was trained with
print(f"\n3. Model Feature Columns ({len(dt_model.feature_columns)} total):")
print(f"   {dt_model.feature_columns[:20]}...")  # Show first 20

# 4. Test prediction on a simple case
print("\n4. Simple Prediction Test:")
print("   Creating test data: Z=92, A=235, MT=18, Energy=[1, 10, 100, 1000]")

# Build test features manually
test_features = []
for energy in [1.0, 10.0, 100.0, 1000.0]:
    feat_dict = {'Z': 92, 'A': 235, 'Energy': energy}

    # Add all MT one-hot columns
    for col in dt_model.feature_columns:
        if col.startswith('MT_'):
            try:
                mt_value = int(col.split('_')[1])
                feat_dict[col] = 1.0 if mt_value == 18 else 0.0
            except:
                feat_dict[col] = 0.0
        elif col not in feat_dict:
            feat_dict[col] = 0.0

    test_features.append(feat_dict)

test_df = pd.DataFrame(test_features)[dt_model.feature_columns]

print("\n   Test DataFrame:")
print(test_df[['Z', 'A', 'Energy']].to_string())

# Make predictions
test_predictions = dt_model.predict(test_df)

print("\n   Predictions:")
for i, (energy, pred) in enumerate(zip([1.0, 10.0, 100.0, 1000.0], test_predictions)):
    print(f"   Energy {energy:8.1f} eV → σ = {pred:10.2f} barns")

# Check if predictions are all identical
unique_preds = len(set(test_predictions.round(6)))
print(f"\n   Unique prediction values: {unique_preds}")
if unique_preds == 1:
    print("   ❌ PROBLEM: All predictions are identical!")
    print("   → This explains the horizontal lines")
else:
    print(f"   ✓ Predictions vary ({unique_preds} unique values)")

# 5. Check if this is a naive vs physics mode issue
print("\n5. Checking Feature Modes:")
print(f"   Naive mode features: Z, A, Energy, MT_onehot")
print(f"   Physics mode features: Z, A, N, log10(Energy), Q, Threshold, ΔZ, ΔA, MT")
print(f"\n   Current training mode appears to be: ", end="")
if 'Q_Value' in dt_model.feature_columns:
    print("PHYSICS")
    print("   ⚠️  Note: Physics mode uses log10(Energy), not linear Energy!")
else:
    print("NAIVE")
    print("   ✓ Using linear Energy values")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
