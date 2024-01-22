package webreduce.extension.classification;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import com.google.common.base.Optional;

import webreduce.extraction.mh.tools.TableConvert;

public class TableParser {
  private TableConvert tableConverter;
  private static final int TABLE_MIN_ROWS = 2;
  private static final int TABLE_MIN_COLS = 2;
  public TableParser() {
    tableConverter = new TableConvert(TABLE_MIN_ROWS, TABLE_MIN_COLS);
  }

  public Element[][] parseTableHTML(String tableHTML) throws TableParsingException {
    return parseTableHTML(tableHTML, true);
  }

  public Element[][] parseTableHTML(String tableHTML, boolean skipSubTables) throws TableParsingException {
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
