import functions.*;

import java.util.ArrayList;
import java.util.Random;

public class GeneticAlgorithm {
    ContinuousFunction function;

    int populationSize = 100;
    double selectionPressure = 0.1;
    double mutationRate = 0.6; // 0.1    0.001
    double mutationMagnitude = 0.1;

    ArrayList<ArrayList<Double>> population;
    ArrayList<Double> populationFitness;

    public GeneticAlgorithm(int function) {
        population = new ArrayList<>();
        populationFitness = new ArrayList<>();

        switch (function){
            case 1 -> this.function = new Ackley();
            case 2 -> this.function = new Griewank();
            case 3 -> this.function = new Rosenbrock();
            case 4 -> this.function = new Spherical();
            case 5 -> this.function = new Whitley();
        }
    }

    public void run() {
        generatePopulation(); // initial random population
        // Check population fitness
        populationFitness.clear();
        for(ArrayList<Double> individual : population)
            populationFitness.add(checkFitness(individual));

        // Several generation
        int genCount = 0;
        double avgfitness = getAvgFitness();
        while(avgfitness > 1 && genCount < 10000) {
            System.out.println(genCount++ + " fitness = " + avgfitness);

            // Create next gen
            ArrayList<ArrayList<Double>> children = new ArrayList<>();
            children.add(findFittestIndividual(population, populationFitness)); // Elitism
            for (int i = 1; i < populationSize; i++) {
                ArrayList<Double> parent1 = tournamentSelection(selectionPressure);
                ArrayList<Double> parent2 = tournamentSelection(selectionPressure);
                children.add(uniformCrossover(parent1, parent2));
            }

            population = children; // Move to next gen

            // Check population fitness
            populationFitness.clear();
            for(ArrayList<Double> individual : population)
                populationFitness.add(checkFitness(individual));

            avgfitness = getAvgFitness();
        }
        System.out.println(genCount + " fitness = " + avgfitness);
    }

    // Generate random individuals as the initial population
    private void generatePopulation() {
        Random random = new Random();
        double gene;
        ArrayList<Double> individual;

        for (int x = 0; x < populationSize; x++) {
            individual = new ArrayList<>(); // Create new individual
            // Generate new individual's genes
            for (int y = 0; y < function.getDimension(); y++) {
                do {
                    gene = (random.nextDouble() * 2 - 1) * 5; // randomly generate new gene
                } while (gene == 0);
                individual.add(gene); // add valid gene to individual
            }
            population.add(individual); // add individual to population
        }
    }

    // Calculate the fitness of an individual
    private double checkFitness(ArrayList<Double> individual) {
        return function.evaluate(individual);
    }

    // Select the single best parent from a number of random individuals in the population
    private ArrayList<Double> tournamentSelection(double selectionPressure) {
        Random rand = new Random();
        ArrayList<ArrayList<Double>> tournamentPopulation = new ArrayList<>();
        ArrayList<Double> tournamentFitness = new ArrayList<>();

        // Select random tournament contestant
        int tournamentSize = (int)(populationSize*selectionPressure);
        for(int i = 0; i < tournamentSize; i++) {
            int pos = rand.nextInt(populationSize);
            tournamentPopulation.add(population.get(pos));
            tournamentFitness.add(populationFitness.get(pos));
        }

        return findFittestIndividual(tournamentPopulation, tournamentFitness);
    }

    private ArrayList<Double> findFittestIndividual(ArrayList<ArrayList<Double>> population, ArrayList<Double> populationFitness) {
        double minFitness = Double.MAX_VALUE;
        ArrayList<Double> fittestIndividual = new ArrayList<>();

        for(int i = 0; i < population.size(); i++) {
            double fitness = populationFitness.get(i);
            if (fitness < minFitness) {
                minFitness = fitness;
                fittestIndividual = population.get(i);
            }
        }

        return fittestIndividual;
    }

    // Use uniform crossover to produce a child from 2 parents
    private ArrayList<Double> uniformCrossover(ArrayList<Double> parent1, ArrayList<Double> parent2) {
        Random rand = new Random();
        ArrayList<Double> child = new ArrayList<>();

        for (int i = 0; i < function.getDimension(); i++){
            if (rand.nextDouble() <= 0.5)
                child.add(parent1.get(i) + mutation());
            else
                child.add(parent2.get(i) + mutation());
        }

        return child;
    }

    // Generates a random mutation value based on the mutationRate and mutationMagnitude
    private double mutation() {
        Random rand = new Random();
        if (rand.nextDouble() <= mutationRate)
            return rand.nextGaussian() * mutationMagnitude;

        return 0;
    }

    // Calculate the average fitness of the whole population
    private double getAvgFitness() {
        double totalFitness = 0;
        for(Double fitness : populationFitness)
            totalFitness += fitness;

        return totalFitness/populationSize;
    }

}
