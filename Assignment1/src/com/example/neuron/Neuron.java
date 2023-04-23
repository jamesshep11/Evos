package com.example.neuron;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/**
 * A single neuron which takes in 3 int values and outputs 1 int value
 */
public class Neuron {
    // The weighting given to each input value ()to be tunes during training
    private ArrayList<Double> weights;

    public Neuron() {
    }

    /**
     * Processes data after the neuron has been trained (does not learn/improve)
     *
     * @param pattern Array of 3 int values as input to the neuron
     * @return Int value after evaluating input
     */
    public int run(ArrayList<Integer> pattern){
        int sum = 0;

        for (int i = 0; i < pattern.size(); i++)
            sum += weights.get(i) * pattern.get(i);

        sum += weights.get(weights.size()-1);

        //System.out.println(sum);

        return sum;
    }

    /**
     * Gradient descent training algorithm initializes weights to random (small) values and then repeatedly adjust
     * weights according to: weight - (n * (-2 * (expectedResult - actualResult) * (inputValue)))
     *
     * @param patterns List of input patterns
     * @param results List of expected output values
     */
    public void gradientDescent(HashMap<Integer, ArrayList<Integer>> patterns, ArrayList<Integer> results){
        Random random = new Random();
        weights = new ArrayList<>();

        // Initialise weights to small random values
        for (int i = 0; i < patterns.get(0).size(); i++)
            weights.add(random.nextDouble() * (Math.pow(-1,random.nextInt(2))));

        weights.add(random.nextDouble() * (Math.pow(-1, random.nextInt(2)))); // bias weight

        for (double w : weights)
            System.out.print(w + ", ");

        System.out.println();

        // Start training
        double SSE = Double.MAX_VALUE; // performance measure (Sum of Squared Errors)
        int count = 0;
        double n = 0.0001; // learning rate

        // While error is too high
        while (SSE > 250) {
            SSE = 0;

            // for each pattern
            for (int p = 0; p < weights.size(); p++) {
                ArrayList<Integer> pattern = patterns.get(p);
                int expected = results.get(p);
                //int diff = expected - run(pattern);

                // adjust each weight (except bias)
                int i;
                for (i = 0; i < weights.size()-1; i++)
                    weights.set(i, weights.get(i) - (n * (-2 * (expected - run(pattern)) * pattern.get(i))));

                // adjust bias weight
                weights.set(i, weights.get(i) - (n * (-2 * (expected - run(pattern)))));

                // Calculate Sum Squared Error
                SSE += Math.pow(expected - run(pattern),2);
            }

            count++;
            System.out.println(count + " iterations. SSE = " + SSE);
        }

        // Output final performance
        System.out.println("SSE = " + SSE);

        // Output final (trained) weights
        for (double w : weights)
            System.out.print(w + ", ");
    }
}
