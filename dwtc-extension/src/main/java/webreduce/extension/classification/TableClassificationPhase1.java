package webreduce.extension.classification;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import com.google.inject.name.Named;
import org.jsoup.nodes.Element;
import webreduce.data.TableType;
import webreduce.extraction.mh.TableClassification;
import webreduce.extraction.mh.features.FeaturesP1;
import webreduce.extraction.mh.tools.ClassificationResult;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import webreduce.extension.classification.TableParser.TableParsingException;
import webreduce.extension.classification.TableParser.TableParsingSubTablesException;

public class TableClassificationPhase1 {

    private FeaturesP1 phase1Features;
    private Classifier classifier1;
    private Attribute classAttr1;
    private double layoutVal, relationVal;
    private TableParser tableParser;

    public TableClassificationPhase1(@Named("phase1ModelPath") String phase1ModelPath) {
        phase1Features = new FeaturesP1();
        tableParser = new TableParser();

        try {
            classifier1 = TableClassification.loadModelFromFile(phase1ModelPath);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Phase 1
        classAttr1 = new Attribute("class", phase1Features.getClassVector());
        layoutVal = classAttr1.indexOfValue("LAYOUT");
        relationVal = classAttr1.indexOfValue("RELATION");
    }

    public ClassificationResult classifyTable(Element[][] convertedTable) {
        double[] dist1;
        Instance currentInst = phase1Features.computeFeatures(convertedTable);
        try {
            double cls = classifier1.classifyInstance(currentInst);
            dist1 = classifier1.distributionForInstance(currentInst);

            if (cls == layoutVal) {
                return new ClassificationResult(TableType.LAYOUT, dist1, null);
            } else if (cls == relationVal) {
                return new ClassificationResult(TableType.RELATION, dist1, null);
            } else {
                throw new Exception("Failure, unexpected classification result.");
            }
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public ClassificationResult classifyTable(String tableHTML) {
        Element[][] table;
        try {
            table = tableParser.parseTableHTML(tableHTML);
        } catch (TableParsingSubTablesException e) {
            System.out.println(e.getMessage());
            return new ClassificationResult(TableType.LAYOUT, new double[] {}, null);
        } catch (TableParsingException e) {
            System.out.println(e.getMessage());
            return null;
        }
        return classifyTable(table);
    }

    public double[] computeFeatures(String tableHTML) {
        Element[][] table;

        try {
            table = tableParser.parseTableHTML(tableHTML);
        } catch (TableParsingException e) {
            System.out.println(e.getMessage());
            return null;
        }

        Instance currentInst = phase1Features.computeFeatures(table);
        System.out.println(Arrays.toString(currentInst.toDoubleArray()));
        return currentInst.toDoubleArray();
    }

      public static void main(String[] args) throws IOException {
    BufferedReader br = new BufferedReader(
        new FileReader("/Users/yuvalpeleg/My Drive/Projects/JParser/tables/table.html"));
    try {
      StringBuilder sb = new StringBuilder();
      String line = br.readLine();

      while (line != null) {
        sb.append(line);
        sb.append(System.lineSeparator());
        line = br.readLine();
      }
      String everything = sb.toString();
      TableClassificationPhase1 classificationPhase1 = new TableClassificationPhase1("/Users/yuvalpeleg/projects/web-table-classification/runtime_testing/resources/RandomForest_P1.mdl");
      var res = classificationPhase1.classifyTable(everything);
      System.out.println(res.toString());

    } catch (Exception e) {
      System.out.println(e.toString());
    } finally {
      br.close();
    }
  }
}
