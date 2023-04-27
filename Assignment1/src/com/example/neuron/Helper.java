package com.example.neuron;

import java.util.ArrayList;
import java.util.Random;

/**
 * A helper class containing generic computational and formatting functions
 */
public class Helper {

    /**
     * Returns an ArrayList of random small values between -1 and 1
     *
     * @param size The size of the ArrayList to be returned
     * @return ArrayList of random small values
     */
    public ArrayList<Double> generateRandomValues(int size) {
        Random random = new Random();
        ArrayList<Double> weights = new ArrayList<>();

        // Initialise weights to small random values
        for (int i = 0; i < size; i++)
            weights.add(random.nextDouble() * (Math.pow(-1,random.nextInt(2))));

        return weights;
    }

    /**
     * Calculates the weight changes using gradient descent algorithm and then calculates the new weight
     * based on the current weight and the learning rate
     *
     * @param inputValue The initial input value
     * @param weight The current weight value
     * @param expected The the expected output value
     * @param actual The actual output value received
     * @param n The learning rate
     * @return The new weight after adjustments
     */
    public double getNewWeight(double inputValue, double weight, double expected, double actual, double n) {
        double change = gradientDescent(inputValue, expected, actual);

        return weight - (n * change);
    }

    /**
     * Calculates the amount of change required to a value using the Gradient Descent algorithm
     *
     * @param inputValue The initial input value
     * @param expected The the expected output value
     * @param actual The actual output value received
     * @return The change required to correct the difference between expected and actual
     */
    public double gradientDescent(double inputValue, double expected, double actual) {
        return -2 * (expected - actual) * inputValue;
    }
}
