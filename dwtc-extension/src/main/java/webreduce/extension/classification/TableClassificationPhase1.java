package webreduce.extension.classification;

import java.util.Arrays;

import com.google.common.base.Optional;
import com.google.inject.name.Named;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import webreduce.data.TableType;
import webreduce.extraction.mh.TableClassification;
import webreduce.extraction.mh.features.FeaturesP1;
import webreduce.extraction.mh.tools.ClassificationResult;
import webreduce.extraction.mh.tools.TableConvert;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;

public class TableClassificationPhase1 {

    private TableConvert tableConverter;
    private FeaturesP1 phase1Features;
    private Classifier classifier1;
    private Attribute classAttr1;
    private double layoutVal, relationVal;
    private static final int TABLE_MIN_ROWS = 2;
    private static final int TABLE_MIN_COLS = 2;

    public TableClassificationPhase1(@Named("phase1ModelPath") String phase1ModelPath) {
        phase1Features = new FeaturesP1();
        tableConverter = new TableConvert(TABLE_MIN_ROWS, TABLE_MIN_COLS);

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
            table = parseTableHTML(tableHTML);
        } catch (TableParsingSubTablesException e) {
            System.out.println(e.getMessage());
            return new ClassificationResult(TableType.LAYOUT, new double[]{}, null);
        } catch (TableParsingException e) {
            System.out.println(e.getMessage());
            return null;
        }
        return classifyTable(table);
    }

    public double[] computeFeatures(String tableHTML) {
        Element[][] table;

        try {
            table = parseTableHTML(tableHTML);
        } catch (TableParsingException e) {
            System.out.println(e.getMessage()); 
            return null;
        }

        Instance currentInst = phase1Features.computeFeatures(table);
        System.out.println(Arrays.toString(currentInst.toDoubleArray()));
        return currentInst.toDoubleArray();
    }

    private Element[][] parseTableHTML(String tableHTML) throws TableParsingException {
        return parseTableHTML(tableHTML, true);
    }

    private Element[][] parseTableHTML(String tableHTML, boolean skipSubTables) throws TableParsingException {
        Document doc = Jsoup.parse(tableHTML);
        Element table = doc.select("table").first();
        if (table == null) {
            throw new TableParsingException("Failure, no table was detected in HTML. Skipping table classification.");
        }

        Elements subTables = table.getElementsByTag("table");
        subTables.remove(table);
        if (subTables.size() > 0 && skipSubTables) {
            throw new TableParsingSubTablesException(
                    "Failure, table includes sub-table(s). Skipping table classification.");
        }

        Optional<Element[][]> convertedTable = tableConverter.toTable(table);
        if (!convertedTable.isPresent()) {
            throw new TableParsingException("toTable() failed. Skipping table classification.");
        }

        return convertedTable.get();
    }

    public class TableParsingException extends Exception {
        private static final long serialVersionUID = 5471172109211007529L;

        public TableParsingException(String errorMessage) {
            super(errorMessage);
        }
    }

    public class TableParsingSubTablesException extends TableParsingException {
        private static final long serialVersionUID = -4415254026083906516L;

        public TableParsingSubTablesException(String errorMessage) {
            super(errorMessage);
        }
    }
}
