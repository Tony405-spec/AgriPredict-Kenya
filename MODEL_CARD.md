# AgriPredict Kenya Model Card

## Model Purpose

AgriPredict Kenya is a Streamlit prototype for estimating crop yield using generated Kenyan farming scenarios and a scikit-learn regression workflow.

## Intended Use

- Learning and portfolio demonstrations.
- Exploring how soil, weather, crop, county, fertilizer, and altitude features affect generated yield estimates.
- Demonstrating Streamlit, SQLite, and machine-learning integration.

## Not Intended For

- Production agronomic advice.
- Farmer financial, insurance, planting, or food-security decisions.
- Deployment without validation on real, representative, locally approved datasets.

## Training Data

The current model workflow generates synthetic records in code. Features include crop type, county, soil pH, soil moisture, fertilizer usage, temperature, rainfall, and altitude.

Metrics from generated data should not be presented as real-world model accuracy.

## Evaluation

The app computes model metrics such as R2, MAE, RMSE, and cross-validation scores. Report these metrics with the generated-data limitation and the input ranges used to produce training samples.

## Limitations

- Generated data cannot represent all real Kenyan farming conditions.
- County defaults simplify weather and altitude variation.
- The model does not account for pests, disease, seed variety, market incentives, irrigation quality, or management practices.
- Predictions may be unreliable outside generated training ranges.

## Responsible Use

Use predictions as educational estimates only. Validate against real agronomic datasets and consult local agricultural experts before using any model output in farmer-facing decisions.
