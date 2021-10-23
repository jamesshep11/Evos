package com.example.neuron;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class Neuron {
    private ArrayList<Double> weights;

    public Neuron() {
    }

    public int run(ArrayList<Integer> pattern){
        int sum = 0;
        for (int i = 0; i < pattern.size(); i++)
            sum += weights.get(i) * pattern.get(i);
        sum += weights.get(weights.size()-1);

        //System.out.println(sum);

        return sum;
    }

    public void gradientDescent(HashMap<Integer, ArrayList<Integer>> patterns, ArrayList<Integer> results){
        // Initialise weights to small random values
        Random random = new Random();
        weights = new ArrayList<>();
        for (int i = 0; i < patterns.get(0).size(); i++)
            weights.add(random.nextDouble() * (Math.pow(-1,random.nextInt(2))));
        weights.add(random.nextDouble() * (Math.pow(-1, random.nextInt(2))));

        for (double w : weights)
            System.out.print(w + ", ");
        System.out.println();

        // Train
        double SSE = Double.MAX_VALUE;
        int count = 0;
        double n = 0.0001;
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
        System.out.println("SSE = " + SSE);
        for (double w : weights)
            System.out.print(w + ", ");
    }
}
