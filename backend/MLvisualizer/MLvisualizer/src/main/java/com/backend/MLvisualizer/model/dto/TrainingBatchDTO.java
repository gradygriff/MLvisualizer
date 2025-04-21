package com.backend.MLvisualizer.model.dto;

import java.util.List;

public class TrainingBatchDTO {
    private List<List<Double>> features;
    private List<Integer> labels;

    public TrainingBatchDTO() {}

    public TrainingBatchDTO(List<List<Double>> features, List<Integer> labels) {
        this.features = features;
        this.labels = labels;
    }

    public List<List<Double>> getFeatures() {
        return features;
    }

    public void setFeatures(List<List<Double>> features) {
        this.features = features;
    }

    public List<Integer> getLabels() {
        return labels;
    }

    public void setLabels(List<Integer> labels) {
        this.labels = labels;
    }
}
