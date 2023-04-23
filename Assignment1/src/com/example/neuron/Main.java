package com.example.neuron;

import java.io.*;
import java.nio.file.ClosedWatchServiceException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class Main {

    static HashMap<Integer, ArrayList<Integer>> patterns = new HashMap<>();
    static ArrayList<Integer> results = new ArrayList<>();

    public static void main(String[] args) throws IOException {
        BufferedReader console = new BufferedReader(
                new InputStreamReader(System.in));

        // Get training data
        System.out.println("Enter training file:");
        String fileName = console.readLine();
        readFile(fileName);

        // Train neuron using gradient decent method
        Neuron neuron = new Neuron();
        neuron.gradientDescent(patterns, results);

        System.out.println("Training complete \n");

        patterns.clear();
        results.clear();

        // Get test data
        System.out.println("Enter test file:");
        fileName = console.readLine();
        readFile(fileName);

        // Use neuron to process test data and evaluate accuracy of neuron after training
        double SSE = 0;
        for (int i = 0; i < patterns.size(); i++) {
            int mark = neuron.run(patterns.get(i));
            SSE += Math.pow(results.get(i) - mark,2);
            System.out.println(results.get(i) + " - " + mark);
        }
        System.out.println("SSE = " + SSE);

        patterns.clear();
        results.clear();

        // Get data to be evaluated
        System.out.println("Enter evaluation file:");
        fileName = console.readLine();
        readFile(fileName);

        // Evaluate input data with trained neuron
        for (int i = 0; i < patterns.size(); i++) {
            int mark = neuron.run(patterns.get(i));
            System.out.println(mark);
        }
    }

    /**
     * Reads data from file and extracts data into input arrays and answers (if training data)
     *
     * @param fileName Name of input file
     */
    public static void readFile(String fileName) {
        try {
            File file = new File(fileName);
            Scanner reader = new Scanner(file);

            int patternCount = 0;

            while (reader.hasNextLine()) {
                // Read line from file
                String line = reader.nextLine();
                String[] patternString = line.split(",");
                ArrayList<Integer> pattern = new ArrayList<>();

                // Extract line into array of int
                for (int i = 0; i < 3; i++)
                    pattern.add(Integer.parseInt(patternString[i]));

                // Add array to list of input data
                patterns.put(patternCount++, pattern);

                // Add 4th element as correct answer (if training data)
                if (patternString.length == 4)
                    try {
                        results.add(Integer.parseInt(patternString[3]));
                    } catch (Exception ignored){ }
            }

            reader.close();
        } catch (FileNotFoundException e) {
            System.out.println("Error reading from file.");
            e.printStackTrace();
        }
    }
}
