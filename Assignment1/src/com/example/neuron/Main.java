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

        System.out.println("Enter training file:");
        String fileName = console.readLine();
        readFile(fileName);

        Neuron neuron = new Neuron();
        neuron.gradientDescent(patterns, results);

        System.out.println("Training complete \n");

        patterns.clear();
        results.clear();

        System.out.println("Enter test file:");
        fileName = console.readLine();
        readFile(fileName);

        double SSE = 0;
        for (int i = 0; i < patterns.size(); i++) {
            int mark = neuron.run(patterns.get(i));
            SSE += Math.pow(results.get(i) - mark,2);
            System.out.println(results.get(i) + " - " + mark);
        }
        System.out.println("SSE = " + SSE);

        patterns.clear();
        results.clear();

        System.out.println("Enter evaluation file:");
        fileName = console.readLine();
        readFile(fileName);

        for (int i = 0; i < patterns.size(); i++) {
            int mark = neuron.run(patterns.get(i));
            System.out.println(mark);
        }
    }

    public static void readFile(String fileName) {
        try {
            File file = new File(fileName);
            Scanner reader = new Scanner(file);

            int patternCount = 0;
            while (reader.hasNextLine()) {
                String line = reader.nextLine();
                String[] patternString = line.split(",");
                ArrayList<Integer> pattern = new ArrayList<>();
                for (int i = 0; i < 3; i++)
                    pattern.add(Integer.parseInt(patternString[i]));
                patterns.put(patternCount++, pattern);
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
