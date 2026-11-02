"""prediction.output tree."""

from kairos.prediction.suffix import SuffixForest


class CurrentCallForest(SuffixForest):
    schema = "current-call-output-distribution-v1"

    def predict_features(self, values):
        result = super().predict_features(values)
        result["source"] = "honest_cdf_current_output"
        return result
