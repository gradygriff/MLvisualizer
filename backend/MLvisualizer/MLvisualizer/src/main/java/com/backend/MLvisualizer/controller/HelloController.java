package com.backend.MLvisualizer.controller;

import com.backend.MLvisualizer.model.dto.TrainingBatchDTO;
import com.backend.MLvisualizer.service.MLServiceClient;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.util.List;
import java.util.Map;

@RestController
public class HelloController {

    MLServiceClient mlServiceClient = new MLServiceClient();

    @GetMapping("/")
    public String hello() {
        return "Hello from MLVisualizer!";
    }

    @PostMapping("/train_logistic")
    public ResponseEntity<Map<String, Object>> trainLogistic(@RequestBody TrainingBatchDTO trainingData) {
        List<List<Double>> features = trainingData.getFeatures();
        List<Integer> labels = trainingData.getLabels();

        // Call Python ML service and get result (including images)
        String resultJson = mlServiceClient.trainLogisticModel(features, labels); // This should return a JSON string

        // Parse the result string into a Map
        ObjectMapper mapper = new ObjectMapper();
        try {
            Map<String, Object> parsed = mapper.readValue(resultJson, Map.class);
            return ResponseEntity.ok(parsed);
        } catch (IOException e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("message", "Error parsing ML response"));
        }
    }
    @PostMapping("/train_linear")
    public ResponseEntity<Map<String, Object>> trainLinear(@RequestBody TrainingBatchDTO trainingData) {
        List<List<Double>> features = trainingData.getFeatures();
        List<Integer> labels = trainingData.getLabels();

        // Call Python ML service and get result (including images)
        String resultJson = mlServiceClient.trainLinearModel(features, labels); // This should return a JSON string

        // Parse the result string into a Map
        ObjectMapper mapper = new ObjectMapper();
        try {
            Map<String, Object> parsed = mapper.readValue(resultJson, Map.class);
            return ResponseEntity.ok(parsed);
        } catch (IOException e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("message", "Error parsing ML response"));
        }
    }

}