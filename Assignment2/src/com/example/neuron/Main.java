package com.example.neuron;

import java.io.*;
import java.util.*;
import java.util.stream.Stream;

public class Main {

    static int numInputs = 15 * 7;
    static int numHiddenNeurons = 1;
    static int numOutputs = 5;

    static HashMap<Integer, int[]> patterns = new HashMap<>();
    static double[][] v = new double[numHiddenNeurons][numInputs+1];
    static double[][] w = new double[numOutputs][numHiddenNeurons+1];
    static HashMap<Integer, int[]> results = new HashMap<>();

    static HashMap<Character, int[]> alpha = new HashMap<>() {{
        put('A', new int[] {0, 0, 0, 0, 1});
        put('B', new int[] {0, 0, 0, 1, 0});
        put('C', new int[] {0, 0, 0, 1, 1});
        put('D', new int[] {0, 0, 1, 0, 0});
        put('E', new int[] {0, 0, 1, 0, 1});
        put('F', new int[] {0, 0, 1, 1, 0});
        put('G', new int[] {0, 0, 1, 1, 1});
        put('H', new int[] {0, 1, 0, 0, 0});
        put('I', new int[] {0, 1, 0, 0, 1});
        put('J', new int[] {0, 1, 0, 1, 0});
        put('K', new int[] {0, 1, 0, 1, 1});
        put('L', new int[] {0, 1, 1, 0, 0});
        put('M', new int[] {0, 1, 1, 0, 1});
        put('N', new int[] {0, 1, 1, 1, 0});
        put('O', new int[] {0, 1, 1, 1, 1});
        put('P', new int[] {1, 0, 0, 0, 0});
        put('Q', new int[] {1, 0, 0, 0, 1});
        put('R', new int[] {1, 0, 0, 1, 0});
        put('S', new int[] {1, 0, 0, 1, 1});
        put('T', new int[] {1, 0, 1, 0, 0});
        put('U', new int[] {1, 0, 1, 0, 1});
        put('V', new int[] {1, 0, 1, 1, 0});
        put('W', new int[] {1, 0, 1, 1, 1});
        put('X', new int[] {1, 1, 0, 0, 0});
        put('Y', new int[] {1, 1, 0, 0, 1});
        put('Z', new int[] {1, 1, 0, 1, 0});
    }};

    public static void main(String[] args) throws IOException {

        // Training
        readFile("TrainingData.txt");
        train();

        //region Validation
        patterns.clear();
        results.clear();
        readFile("ValidationData.txt");

        double SSE = 0;
        for (int p = 0; p < patterns.size(); p++) {
            int[] o = run(patterns.get(p));
            double E = 0;
            for (int k = 0; k < o.length; k++)
                E += Math.pow(results.get(p)[k] - o[k],2);
            SSE += 0.5 * E;
            System.out.println(Arrays.toString(results.get(p)) + " - " + Arrays.toString(o));
        }
        System.out.println("SSE = " + SSE);
        //endregion

        //region Testing
        patterns.clear();
        results.clear();
        readFile("TestData.txt");

        for (int i = 0; i < patterns.size(); i++) {
            int[] o = run(patterns.get(i));
            System.out.println(Arrays.toString(o));
        }
        //endregion
    }

    public static void train() {
        //region Initialise weights
        Random random = new Random();
        for (double[] x : v)
            for (double y : x){
                do {
                    y = random.nextDouble() * (Math.pow(-1, random.nextInt(2)));
                } while (y == 0);
            }
        for (double[] x : w)
            for (double y : x){
                do {
                    y = random.nextDouble() * (Math.pow(-1, random.nextInt(2)));
                } while (y == 0);
            }
        //endregion

        // Train
        double SSE = Double.MAX_VALUE;
        int count = 0;
        double n = 0.0001;
        while (SSE > 250) {
            SSE = 0;
            // for each pattern
            for (int p = 0; p < patterns.size()-1; p++) {
                int[] z = patterns.get(p);
                int[] expected = results.get(p);
                // adjust each w
                for (int k = 0; k < w.length; k++)
                    for (int j = 0; j < w[k].length; j++) {
                        int[] o = run(z);
                        int y = y(z, j);
                        w[k][j] -= n * (-2 * (expected[k] - o[k]) * o[k] * (1 - o[k]) * y);
                    }
                // adjust each v
                for (int j = 0; j < v.length; j++)
                    for (int i = 0; i < v[j].length; i++) {
                        int[] o = run(z);
                        int y = y(z, j);
                        double diff = 0;
                        for (int k = 0; k < w.length; k++)
                            diff += -2 * (expected[k] - o[k]) * o[k] * (1 - o[k]) * w[k][j] * y * (1 - y) * z[i];
                        v[j][i] -= n * diff;
                    }

                // Calculate Sum Squared Error
                int[] o = run(z);
                double E = 0;
                for (int x = 0; x < o.length; x++)
                    E += Math.pow(expected[x] - o[x],2);
                SSE += 0.5 * E;
            }
            count++;
            System.out.println(count + " iterations. SSE = " + SSE);
        }

        System.out.println("Avg. SSE = " + SSE/count);
    }

    public static int y(int[] z, int y) {
        int sum = 0;
        for (int i = 0; i < z.length; i++)
            sum += v[y][i] * z[i];

        return sum;
    }

    public static int[] run(int[] z) {
        int[] o = new int[numOutputs];

        // For each output neuron
        for (int k = 0; k < o.length; k++) {
            double sum = 0; // output sum
            double sum2;    // hidden output sum
            // for each hidden neuron
            for (int j = 0; j < w.length; j++) {
                sum2 = 0;
                // for each input
                for (int i = 0; i < v.length; i++)
                    sum2 += v[j][i] * z[i];

                sum += w[k][j] * sum2;
            }
            o[k] = Math.round((float)sum); // round because we only want 0 or 1
        }

        return o;
    }

    public static void readFile(String fileName) {
        try {
            File file = new File(fileName);
            Scanner reader = new Scanner(file);

            int patternCount = 0;
            while (reader.hasNextLine()) {
                // Get result
                StringBuilder line = new StringBuilder(reader.nextLine());
                results.put(patternCount, alpha.get(line.toString()));

                // Get pattern as string
                line = new StringBuilder(reader.nextLine());
                for (int i = 0; i < 14; i++)
                    line.append(",").append(reader.nextLine());
                line.append(",1"); // bias input

                // Convert pattern to int[]
                String[] patternString = line.toString().split(",");
                int[] pattern = new int[numInputs + 1];
                for (int i = 0; i < patternString.length; i++)
                    pattern[i] = Integer.parseInt(patternString[i]);

                patterns.put(patternCount++, pattern);
            }
            reader.close();
        } catch (FileNotFoundException e) {
            System.out.println("Error reading from file.");
            e.printStackTrace();
        }
    }
}
