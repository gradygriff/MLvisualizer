package com.backend.MLvisualizer.service;

import org.springframework.http.*;
import org.springframework.web.client.RestTemplate;
import java.util.*;

public class MLServiceClient {

    private final RestTemplate restTemplate = new RestTemplate();
    private final String flaskApiUrl = "http://localhost:5000";

    public String trainLogisticModel(List<List<Double>> features, List<Integer> labels) {
        String url = flaskApiUrl + "/train_logistic";
        Map<String, Object> request = new HashMap<>();
        request.put("features", features);
        request.put("labels", labels);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(request, headers);

        ResponseEntity<String> response = restTemplate.postForEntity(url, entity, String.class);
        return response.getBody();
    }

    public String trainLinearModel(List<List<Double>> features, List<Integer> labels) {
        String url = flaskApiUrl + "/train_linear";
        Map<String, Object> request = new HashMap<>();
        request.put("features", features);
        request.put("labels", labels);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(request, headers);

        ResponseEntity<String> response = restTemplate.postForEntity(url, entity, String.class);
        return response.getBody();
    }
    public List<Integer> predict(List<List<Double>> features) {
        String url = flaskApiUrl + "/predict";
        Map<String, Object> request = new HashMap<>();
        request.put("features", features);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(request, headers);

        ResponseEntity<Map> response = restTemplate.postForEntity(url, entity, Map.class);
        return (List<Integer>) response.getBody().get("predictions");
    }
}
