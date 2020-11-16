package webreduce.extension.classification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

import webreduce.extraction.mh.tools.ClassificationResult;

public class TableClassificationUtils {
  
  private TableClassificationPhase1 tableClassifier;

  public TableClassificationUtils(String modelPath) {
    tableClassifier = new TableClassificationPhase1(modelPath);
  }

  public void writeFeatures(String htmlTablesFile) throws IOException {
    writeFeatures(htmlTablesFile, "features.csv");
  }
  
  public void writeFeatures(String htmlTablesFile, String featureResultsPath) throws IOException {
    BufferedReader csvReader = new BufferedReader(new FileReader(htmlTablesFile));
    PrintWriter writer = new PrintWriter(new File(featureResultsPath));
    StringBuilder sb = new StringBuilder();
    String row;

    while ((row = csvReader.readLine()) != null) {
      String[] data = row.split(",", 2);
      String id = data[0];
      String tableHTML = data[1];

      double[] features = tableClassifier.computeFeatures(tableHTML);

      // do not include tables for which features cannot be computed
      if (features == null) {
        continue;
      }

      sb.append(id);
      for (double feature : features) {
        sb.append(",");
        sb.append(feature);
      }
      sb.append('\n');
    }

    writer.write(sb.toString());
    csvReader.close();
    writer.flush();
    writer.close();
  }

  public void writeClassifications(String htmlTablesFile) throws IOException {
    writeClassifications(htmlTablesFile, "predictions.csv");
  }

  public void writeClassifications(String htmlTablesFile, String predictionResultsPath) throws IOException {
    BufferedReader csvReader = new BufferedReader(new FileReader(htmlTablesFile));
    PrintWriter writer = new PrintWriter(new File(predictionResultsPath));
    StringBuilder sb = new StringBuilder();
    String row;

    while ((row = csvReader.readLine()) != null) {
      String[] data = row.split(",", 2);
      String id = data[0];
      String tableHTML = data[1];

      // Printing out progress
      System.out.println("ID: ");
      System.out.print(id);

      ClassificationResult cResult = tableClassifier.classifyTable(tableHTML);
      String result = cResult == null ? "Failure" : cResult.getTableType().toString();

      sb.append(id);
      sb.append(',');
      sb.append(result);
      sb.append('\n');
    }

    writer.write(sb.toString());
    csvReader.close();
    writer.flush();
    writer.close();
  }

  public void benchmarkTableClassification(String htmlTablesFile) throws IOException {
    long startTime = System.currentTimeMillis();
    BufferedReader csvReader = new BufferedReader(new FileReader(htmlTablesFile));
    String row;

    while ((row = csvReader.readLine()) != null) {
      String tableHTML = row.split(",", 2)[1];
      tableClassifier.classifyTable(tableHTML);
    }

    csvReader.close();
    long endTime = System.currentTimeMillis();
    System.out.println("############ BENCHMARKING RESULT ############");
    System.out.println("||||| DWTC - RF classifier function took " + String.format("%.03f", (endTime - startTime) / 1000.0) + "s");
  }

  public static void main(String[] args) throws IOException {
    String project_dir = System.getProperty("user.dir");
    String htmlTablesDefaultFile = project_dir + "/runtime_testing/resources/performance_testing_1000_tables_formatted.csv"; 
    String modelFile = project_dir + "/runtime_testing/resources/RandomForest_P1.mdl";

    TableClassificationUtils utils = new TableClassificationUtils(modelFile);
    
    // utils.writeClassifications(htmlTablesDefaultFile); 
    utils.benchmarkTableClassification(htmlTablesDefaultFile);
    // utils.writeFeatures(htmlTablesDefaultFile);
  }
}
