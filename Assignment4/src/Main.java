import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) throws IOException {

        /*GeneticAlgorithm algorithm = new GeneticAlgorithm(1);
        algorithm.run();*/

        /*DifferentialEvolution algorithm = new DifferentialEvolution(5);
        algorithm.run();*/

        ParticleSwarm algorithm = new ParticleSwarm(1);
        algorithm.run();
    }
}
