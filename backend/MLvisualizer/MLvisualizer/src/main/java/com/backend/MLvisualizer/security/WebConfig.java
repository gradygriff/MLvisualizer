package com.backend.MLvisualizer.security;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {

    static final String origin = "http://localhost:3000";
    static final String allPattern = "/**";

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping(allPattern).allowedOrigins(origin);
    }
}