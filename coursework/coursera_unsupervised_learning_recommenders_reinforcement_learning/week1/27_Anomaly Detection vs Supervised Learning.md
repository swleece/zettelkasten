<27>

## Title

Compare and Contrast, Feature Selection

## Description

#### Examples

| Anomaly Detection | Supervised Learning |
| --- | --- |
| small number of pos examples, large neg | large number of pos and neg |
| good if many 'types' of anomalies and future anomalies may be different | good if there are enough pos examples that are predictive of future |
| fraud detection | email spam classification |
| manufacturing defects, unseen | manufacturing defects, previously seen |
| monitoring machines in a data center| weather prediction |
| security-related monitoring | disease classification |

## Additional Notes

#### Choosing Features:

- extra important for anomaly detection
- Try to make sure the features are about Gaussian, may be able to massage/transform data to be more gaussian. (log(x), log(x+c), x^(1/2))
- can also use error analysis to update function or feature choices
- also consider using combinations of features as a feature
- e.g. cpu load / network traffic vs each of them individually when monitoring hardware

## Linked Cards

{{ direct link to another card }}

## Tags

[[Machine Learning]]
