import functions.*;

import java.util.ArrayList;
import java.util.Random;

public class DifferentialEvolution {
    ContinuousFunction function;

    int populationSize = 100;
    double mutationRate = 1; // 0.1
    double crossoverRate = 0.5;

    ArrayList<ArrayList<Double>> population;
    ArrayList<Double> populationFitness;

    public DifferentialEvolution(int function) {
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

    // DE\rand\1\bin
    public void run() {
        generatePopulation(); // initial random population
        // Check population fitness
        populationFitness.clear();
        for(ArrayList<Double> individual : population)
            populationFitness.add(checkFitness(individual));

        // Several generation
        Random rand = new Random();
        int genCount = 0;
        double avgFitness = getAvgFitness();
        while(avgFitness > 1 && genCount < 10000) {
            System.out.println(genCount++ + " fitness = " + avgFitness);

            // Create next gen
            ArrayList<ArrayList<Double>> children = new ArrayList<>();
            children.add(findFittestIndividual(population, populationFitness)); // Elitism
            for (int i = 1; i < populationSize; i++) {
                ArrayList<Double> cur = population.get(i);

                // Pick 3 random 'parents'
                ArrayList<Double> x1 = population.get(rand.nextInt(populationSize));
                ArrayList<Double> x2 = population.get(rand.nextInt(populationSize));
                ArrayList<Double> x3 = population.get(rand.nextInt(populationSize));

                // Create mutant vector and trial vector
                ArrayList<Double> mutant = mutantVector(x1, x2, x3);
                ArrayList<Double> trialVec = uniformCrossover(cur, mutant);
                double fitness = checkFitness(trialVec);

                // Compare individual and trial to choose child
                if(fitness < populationFitness.get(i))
                    children.add(trialVec);
                else
                    children.add(cur);
            }

            population = children; // Move to next gen

            // Check population fitness
            populationFitness.clear();
            for(ArrayList<Double> individual : population)
                populationFitness.add(checkFitness(individual));

            avgFitness = getAvgFitness();
        }
        System.out.println(genCount + " fitness = " + avgFitness);
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

    // Generates a mutant vector based on the 3 input vectors
    private ArrayList<Double> mutantVector(ArrayList<Double> x1, ArrayList<Double> x2, ArrayList<Double> x3) {
        ArrayList<Double> diff = vectorDiff(x2, x3);

        ArrayList<Double> mutant = new ArrayList<>();
        for(int i = 0; i < x1.size(); i++)
            mutant.add(x1.get(i) + (mutationRate * diff.get(i)));

        return mutant;
    }


    // Calculate the fitness of an individual
    private double checkFitness(ArrayList<Double> individual) {
        return function.evaluate(individual);
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
        double guarantee = rand.nextInt(function.getDimension());

        for (int i = 0; i < function.getDimension(); i++){
            if (rand.nextDouble() <= crossoverRate || i == guarantee)
                child.add(parent1.get(i));
            else
                child.add(parent2.get(i));
        }

        return child;
    }

    // Calculate the average fitness of the whole population
    private double getAvgFitness() {
        double totalFitness = 0;
        for(Double fitness : populationFitness)
            totalFitness += fitness;

        return totalFitness/populationSize;
    }

    private ArrayList<Double> vectorDiff(ArrayList<Double> v1, ArrayList<Double> v2){
        ArrayList<Double> diff = new ArrayList<>();
        for(int i = 0; i < v1.size(); i++)
            diff.add(v1.get(i) - v2.get(i));

        return diff;
    }

}
