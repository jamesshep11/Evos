package com.example.neuron;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * A single neuron which takes in 3 int values and outputs 1 int value
 */
public class Neuron {
    Helper helper;

    ArrayList<Double> weights; // The weights given to each input value to be tunes during training
    final double n = 0.00001; // learning rate
    final double MinSSE = 250; // Minimum acceptable error (stopping condition for training)
    final int bias = 1;

    /**
     * Construct a neuron and initialise random small weights
     *
     * @param numInputs Number of input parameters given to the neuron
     */
    public Neuron(int numInputs) {
        helper = new Helper();
        weights = helper.generateRandomValues(numInputs+1); // +1 to include the bias

        // Display starting weights
        for (double w : weights)
            System.out.print(w + ", ");

        System.out.println();
    }

    /**
     * Processes data after the neuron has been trained (does not learn/improve)
     *
     * @param pattern Array of 3 int values as input to the neuron
     * @return Int value after evaluating input
     */
    public int run(ArrayList<Integer> pattern){
        int sum = 0;

        // Sum input*weight for each input
        for (int i = 0; i < pattern.size(); i++)
            sum += weights.get(i) * pattern.get(i);

        // Add in the bias
        sum += weights.get(weights.size()-1) * bias;

        return sum;
    }

    /**
     * Trains the node using feed forward and gradient descent (stochastic):
     * new weight = weight - (n * (-2 * (expectedResult - actualResult) * (inputValue)))
     *
     * @param patterns List of input patterns
     * @param expectedResults List of expected output values
     */
    public void train(HashMap<Integer, ArrayList<Integer>> patterns, ArrayList<Integer> expectedResults){
        double SSE; // performance measure (Sum of Squared Errors)
        int iterations = 0;

        // Start training
        do {
            SSE = 0;

            // for each pattern
            for (int p = 0; p < weights.size(); p++) {
                ArrayList<Integer> pattern = patterns.get(p);
                int expected = expectedResults.get(p);
                double actual = run(pattern);

                // adjust each weight (except bias)
                int i;
                for (i = 0; i < weights.size() - 1; i++) {
                    double weightChange = helper.gradientDescent(pattern.get(i), expected, actual);
                    double newWeight = weights.get(i) - (n * weightChange);
                    weights.set(i, newWeight);
                }

                // adjust bias weight
                double weightChange = helper.gradientDescent(bias, expected, actual);
                double newBiasWeight = weights.get(i) - (n * weightChange);
                weights.set(i, newBiasWeight);

                // Calculate Sum Squared Error
                SSE += Math.pow(expected - run(pattern), 2);
            }

            iterations++;
            System.out.println(iterations + " iterations. SSE = " + SSE);
        } while (SSE > MinSSE); // While error is too high

        // Output final performance
        System.out.println("SSE = " + SSE);

        // Output final (trained) weights
        for (double w : weights)
            System.out.print(w + ", ");
    }
}