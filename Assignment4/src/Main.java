import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(
                new InputStreamReader(System.in));

        /*GeneticAlgorithm algorithm = new GeneticAlgorithm(1);
        algorithm.run();*/

        /*DifferentialEvolution algorithm = new DifferentialEvolution(5);
        algorithm.run();*/

        ParticleSwarm algorithm = new ParticleSwarm(5);
        algorithm.run();
    }
}
