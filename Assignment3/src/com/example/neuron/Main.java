package com.example.neuron;

import java.io.*;
import java.lang.reflect.Array;
import java.util.*;

public class Main {

    static int method = 1;
    static int numParams = 7;
    static int populationSize = 100;
    static double selectionPressure = 0.1;
    static double mutationRate = 0.5;
    static double mutationMagnitude = 0.1;

    static HashMap<Integer, int[]> patterns = new HashMap<>();
    static ArrayList<Integer> results = new ArrayList<>();
    static double[][] population = new double[populationSize][numParams];
    static double[] populationFitness = new double[populationSize];

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(
                new InputStreamReader(System.in));

        train();

        System.out.println("Continue?");
        reader.readLine();
        evaluate();

        System.out.println("Continue?");
        reader.readLine();
        test();
    }

    private static void train() {
        patterns.clear();
        results.clear();
        readFile("TrainingData.csv");

        generatePopulation(); // initial random population
        // Check population fitness
        for(int i = 0; i < populationSize; i++)
            populationFitness[i] = checkFitness(population[i]);

        // Several generation
        int genCount = 0;
        double avgSSE = getAvgFitness();
        while(avgSSE > 50 && genCount < 10000) {
            System.out.println(genCount++ + " SSE = " + avgSSE);

            // Create next gen
            double[][] children = new double[populationSize][numParams];
            children[0] = findFittestIndividual(population, populationFitness); // Elitism
            for (int i = 1; i < populationSize; i++) {
                double[] parent1 = tournamentSelection(selectionPressure);
                double[] parent2 = tournamentSelection(selectionPressure);
                children[i] = uniformCrossover(parent1, parent2);
            }

            population = children; // Move to next gen

            // Check population fitness
            for(int i = 0; i < populationSize; i++)
                populationFitness[i] = checkFitness(population[i]);

            avgSSE = getAvgFitness();
        }
        System.out.println(genCount + " SSE = " + avgSSE);
    }

    private static void evaluate() {
        patterns.clear();
        results.clear();
        readFile("ValidationData.csv");

        // Check population fitness against unseen data
        for(int i = 0; i < populationSize; i++)
            populationFitness[i] = checkFitness(population[i]);

        // calculate avgSSE again unseen data
        double avgSSE = getAvgFitness();
        System.out.println("SSE = " + avgSSE);
    }

    private static void test() {
        patterns.clear();
        results.clear();
        readFile("TestData.csv");
        double[] individual = findFittestIndividual(population, populationFitness);

        // for each pattern
        for(int x = 0; x < patterns.size(); x++){
            int[] pattern = patterns.get(x);
            double salary = 0;

            // predict the output using the fittest individual
            switch (method) {
                case 1 -> {
                    for (int i = 0; i < pattern.length; i++)
                        salary += individual[i] * pattern[i];
                }
                case 2 -> {
                    for (int i = 0; i < pattern.length; i++)
                        salary += individual[i] * pattern[i];
                    salary += individual[pattern.length];
                }
                case 3 -> {
                    for (int i = 0; i < pattern.length; i++)
                        salary += Math.pow(individual[i] * pattern[i], individual[i + 8]);
                    salary += individual[pattern.length];
                }
            }

            System.out.println(Math.round(salary));
        }
    }

    // Generate random individuals as the initial population
    private static void generatePopulation() {
        Random random = new Random();

        for (int x = 0; x < populationSize; x++)
            for (int y = 0; y < numParams; y++)
                do {
                    population[x][y] = random.nextGaussian();
                } while (population[x][y] == 0);
    }

    // Calculate the fitness of an individual
    private static double checkFitness(double[] individual) {
        double SSE = 0;
        for(int x = 0; x < patterns.size(); x++){
            int[] pattern = patterns.get(x);
            int expected = results.get(x);
            double actual = 0;

            switch (method) {
                case 1 -> {
                    for (int i = 0; i < pattern.length; i++)
                        actual += individual[i] * pattern[i];
                }
                case 2 -> {
                    for (int i = 0; i < pattern.length; i++)
                        actual += individual[i] * pattern[i];
                    actual += individual[pattern.length];
                }
                case 3 -> {
                    for (int i = 0; i < pattern.length; i++)
                        actual += Math.pow(individual[i] * pattern[i], individual[i + 8]);
                    actual += individual[pattern.length];
                }
            }
            SSE += Math.pow(expected - actual, 2);
        }

        return SSE / patterns.size() / 10000;
    }

    // Select the single best parent from a number of random individuals in the population
    private static double[] tournamentSelection(double selectionPressure) {
        Random rand = new Random();
        double[][] tournamentPopulation = new double[(int)(populationSize*selectionPressure)][numParams];
        double[] tournamentFitness = new double[(int)(populationSize*selectionPressure)];

        // Select random tournament contestant
        for(int i = 0; i < tournamentPopulation.length; i++) {
            int pos = rand.nextInt(populationSize);
            tournamentPopulation[i] = population[pos];
            tournamentFitness[i] = populationFitness[pos];
        }

        return findFittestIndividual(tournamentPopulation, tournamentFitness);
    }

    private static double[] findFittestIndividual(double[][] population, double[] populationFitness) {
        double minFitness = Double.MAX_VALUE;
        double[] fittestIndividual = new double[numParams];

        for(int i = 0; i < population.length; i++) {
            double fitness = populationFitness[i];
            if (fitness < minFitness) {
                minFitness = fitness;
                fittestIndividual = population[i];
            }
        }

        return fittestIndividual;
    }

    // Use uniform crossover to produce a child from 2 parents
    private static double[] uniformCrossover(double[] parent1, double[] parent2) {
        Random rand = new Random();
        double[] child = new double[numParams];

        for (int i = 0; i < numParams; i++){
            if (rand.nextDouble() <= 0.5)
                child[i] = parent1[i] + mutation();
            else
                child[i] = parent2[i] + mutation();
        }

        return child;
    }

    // Generates a random mutation value based on the mutationRate and mutationMagnitude
    private static double mutation() {
        Random rand = new Random();
        if (rand.nextDouble() <= mutationRate)
            return rand.nextGaussian() * mutationMagnitude;

        return 0;
    }

    // Calculate the average fitness of the whole population
    private static double getAvgFitness() {
        double SSE = 0;
        for(int i = 0; i < populationSize; i++)
            SSE += populationFitness[i];

        return SSE/populationSize;
    }

    public static void readFile(String fileName) {
        try {
            File file = new File(fileName);
            Scanner reader = new Scanner(file);
            reader.nextLine(); // skip titles
            boolean testing = fileName.contains("Test");

            int patternCount = 0;
            while (reader.hasNextLine()) {
                // Extract values
                String line = reader.nextLine();
                String[] values = line.split(",");

                // Get result
                if (!testing)
                    results.add(Integer.parseInt(values[0]));

                // Get pattern as string
                int[] pattern = new int[testing ? values.length : values.length - 1];
                int startValue = testing ? 0 : 1; // because test data doesn't start with the result
                for (int i = 0; i < pattern.length; i++)
                    pattern[i] = Integer.parseInt(values[startValue++]);

                patterns.put(patternCount++, pattern);
            }
            reader.close();
        } catch (FileNotFoundException e) {
            System.out.println("Error reading from file.");
            e.printStackTrace();
        }
    }
}
