import functions.*;

import java.util.ArrayList;
import java.util.Random;

public class ParticleSwarm {
    ContinuousFunction function;

    int populationSize = 150;
    double w = 1;
    double c1 = 0.9;
    double c2 = 0.1;

    ArrayList<ArrayList<Double>> population;
    ArrayList<ArrayList<Double>> populationVelocity;
    ArrayList<ArrayList<Double>> personalBests;
    ArrayList<Double> globalBest;
    ArrayList<Double> populationFitness;

    public ParticleSwarm(int function) {
        population = new ArrayList<>();
        populationFitness = new ArrayList<>();
        populationVelocity = new ArrayList<>();
        personalBests = new ArrayList<>();

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
        globalBest = findFittestIndividual(population, populationFitness); // initialize global best

        // Start swarming
        int genCount = 0;
        double avgFitness = getAvgFitness();
        while(avgFitness > 1 && genCount < 1000) {
            System.out.println(genCount++ + " fitness = " + avgFitness);

            // Update population positions
            for (int i = 1; i < populationSize; i++) {
                ArrayList<Double> x = population.get(i); // current individual
                ArrayList<Double> v = populationVelocity.get(i); // current individual's velocity
                ArrayList<Double> y = personalBests.get(i); // current individual's personal best

                // Update personal best
                if (checkFitness(x) < checkFitness(personalBests.get(i)))
                    personalBests.set(i, x);
                // Update global best
                if (checkFitness(personalBests.get(i)) < checkFitness(globalBest))
                    globalBest = personalBests.get(i);

                // Move individual one dimension at a time
                Random rand = new Random();
                for (int j = 0; j < function.getDimension(); j++){
                    double r1 = rand.nextDouble(); // randomizer 1
                    double r2 = rand.nextDouble(); // randomizer 2

                    // Calculate new velocity component and update individual
                    double newVelocity = (w * v.get(j)) + (c1 * r1 * (y.get(j) - x.get(j))) + (c2 * r2 * (globalBest.get(j) - x.get(j)));
                    x.set(j, x.get(j) + newVelocity);
                }
                // Check individual's fitness
                populationFitness.set(i, checkFitness(x));
            }

            avgFitness = getAvgFitness();
        }
        System.out.println(genCount + " fitness = " + avgFitness);
    }

    // Generate random individuals as the initial population
    private void generatePopulation() {
        Random random = new Random();
        double gene;
        ArrayList<Double> individual;
        ArrayList<Double> velocity;

        for (int x = 0; x < populationSize; x++) {
            individual = new ArrayList<>(); // Create new individual
            velocity = new ArrayList<>();
            // Generate new individual's genes
            for (int y = 0; y < function.getDimension(); y++) {
                do {
                    gene = (random.nextDouble() * 2 - 1) * 5; // randomly generate new gene
                } while (gene == 0);
                individual.add(gene); // add valid gene to individual
                velocity.add(0.0); // initialize velocity
            }
            population.add(individual); // add individual to population
            populationVelocity.add(velocity);
        }
        personalBests = (ArrayList<ArrayList<Double>>) population.clone(); // initialize personal bests
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

    // Calculate the average fitness of the whole population
    private double getAvgFitness() {
        double totalFitness = 0;
        for(Double fitness : populationFitness)
            totalFitness += fitness;

        return totalFitness/populationSize;
    }

}
