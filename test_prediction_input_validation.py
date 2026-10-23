from prediction_validation import validate_prediction_input


FEATURE_NAMES = [
    "crop_type",
    "county",
    "soil_ph",
    "soil_moisture",
    "fertilizer_usage",
    "temperature",
    "rainfall",
    "altitude",
]


def valid_input():
    return {
        "crop_type": "Maize",
        "county": "Nakuru",
        "soil_ph": "6.5",
        "soil_moisture": 25,
        "fertilizer_usage": 100,
        "temperature": 20,
        "rainfall": 80,
        "altitude": 1850,
    }


def test_validate_prediction_input_normalizes_numeric_values():
    validated = validate_prediction_input(valid_input(), FEATURE_NAMES)

    assert validated["crop_type"] == "Maize"
    assert validated["soil_ph"] == 6.5
    assert validated["altitude"] == 1850.0


def test_validate_prediction_input_rejects_missing_features():
    payload = valid_input()
    del payload["rainfall"]

    try:
        validate_prediction_input(payload, FEATURE_NAMES)
    except ValueError as exc:
        assert "missing required prediction features: rainfall" in str(exc)
    else:
        raise AssertionError("missing rainfall should raise ValueError")


def test_validate_prediction_input_rejects_non_numeric_values():
    payload = valid_input()
    payload["soil_ph"] = "acidic"

    try:
        validate_prediction_input(payload, FEATURE_NAMES)
    except ValueError as exc:
        assert "soil_ph must be numeric" in str(exc)
    else:
        raise AssertionError("non-numeric soil_ph should raise ValueError")


def test_validate_prediction_input_rejects_non_finite_values():
    payload = valid_input()
    payload["temperature"] = "nan"

    try:
        validate_prediction_input(payload, FEATURE_NAMES)
    except ValueError as exc:
        assert "temperature must be finite" in str(exc)
    else:
        raise AssertionError("non-finite temperature should raise ValueError")
