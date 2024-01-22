package webreduce.extension.classification;

import org.jsoup.nodes.Element;

import webreduce.data.TableType;
import webreduce.extraction.mh.tools.ClassificationResult;
import webreduce.extraction.mh.TableClassification;
import webreduce.extension.classification.TableParser.TableParsingException;
import webreduce.extension.classification.TableParser.TableParsingSubTablesException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class TableClassifier {
  private TableParser tableParser;
  private TableClassification tableClassification;

  public TableClassifier() {
    tableParser = new TableParser();
    tableClassification = new TableClassification(
        "/RandomForest_P1.mdl",
        "/RandomForest_P2.mdl");
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
    return tableClassification.classifyTable(table);
  }

  public static void main(String[] args) throws IOException {
    BufferedReader br = new BufferedReader(
        new FileReader("/Users/yuvalpeleg/My Drive/Projects/JParser/tables/table.html"));
    try {
      TableClassification.loadModelFromClasspath("/RandomForest_P1.mdl");
      StringBuilder sb = new StringBuilder();
      String line = br.readLine();

      while (line != null) {
        sb.append(line);
        sb.append(System.lineSeparator());
        line = br.readLine();
      }
      String everything = sb.toString();
      TableClassifier table_classifier = new TableClassifier();
      var res = table_classifier.classifyTable(everything);
      System.out.println(res.toString());

    } catch (Exception e) {
      System.out.println(e.toString());
    } finally {
      br.close();
    }
  }
}
