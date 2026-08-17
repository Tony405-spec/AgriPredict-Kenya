import math


def validate_prediction_input(input_data, feature_names):
    """Validate and normalize one prediction payload before model scoring."""
    missing_features = [feature for feature in feature_names if feature not in input_data]
    if missing_features:
        raise ValueError(f"missing required prediction features: {', '.join(missing_features)}")

    validated = {}
    categorical_features = {"crop_type", "county"}

    for feature in feature_names:
        value = input_data[feature]
        if feature in categorical_features:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{feature} must be a non-empty string")
            validated[feature] = value
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} must be numeric") from exc

        if not math.isfinite(numeric_value):
            raise ValueError(f"{feature} must be finite")

        validated[feature] = numeric_value

    return validated
