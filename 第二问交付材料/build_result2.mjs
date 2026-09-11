import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [templateArg, temperatureArg, moistureArg, outputArg, previewArg] = process.argv.slice(2);
if (!templateArg || !temperatureArg || !moistureArg || !outputArg || !previewArg) {
  throw new Error("用法：node build_result2.mjs <模板.xlsx> <温度.csv> <水分.csv> <输出.xlsx> <预览目录>");
}

async function readNumericCsv(csvPath) {
  const text = (await fs.readFile(path.resolve(csvPath), "utf8")).replace(/^\uFEFF/, "").trim();
  const lines = text.split(/\r?\n/);
  if (lines.length !== 10802) {
    throw new Error(`${csvPath} 应含1行表头和10801行数据，实际为${lines.length}行。`);
  }
  return lines.map((line, rowIndex) => {
    const cells = line.split(",");
    if (cells.length !== 22) throw new Error(`${csvPath} 第${rowIndex + 1}行不是22列。`);
    return cells.map((cell, columnIndex) => {
      if (rowIndex === 0 && columnIndex === 0) return "时间\\到药材中心的距离/cm";
      const numeric = Number(cell);
      if (!Number.isFinite(numeric)) throw new Error(`${csvPath} 存在非有限数值。`);
      if (rowIndex === 0) return Number.isInteger(numeric) ? String(numeric) : numeric.toFixed(1);
      if (columnIndex === 0) return numeric;
      return Math.round(numeric * 10000) / 10000;
    });
  });
}

const temperatureData = await readNumericCsv(temperatureArg);
const moistureData = await readNumericCsv(moistureArg);
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path.resolve(templateArg)));

for (const [sheetName, data] of [["温度", temperatureData], ["水分浓度", moistureData]]) {
  const sheet = workbook.worksheets.getItem(sheetName);
  sheet.getRange("A1:F5").clear({ applyTo: "contents" });
  sheet.getRange("A1").writeValues(data);
  const fullRange = sheet.getRange("A1:V10802");
  fullRange.format.font = { name: "宋体", size: 10, color: "#222222" };
  fullRange.format.verticalAlignment = "center";
  fullRange.format.horizontalAlignment = "center";
  fullRange.format.rowHeight = 18;
  sheet.getRange("A1:V1").format = {
    fill: "#D9EAF7",
    font: { name: "宋体", size: 10, bold: true, color: "#1F1F1F" },
    horizontalAlignment: "center", verticalAlignment: "center",
    borders: { bottom: { style: "thin", color: "#7F8C8D" } },
  };
  sheet.getRange("A1:A10802").format.columnWidth = 27;
  sheet.getRange("B1:V10802").format.columnWidth = 10;
  sheet.getRange("A2:A10802").format.numberFormat = "0";
  sheet.getRange("B1:V1").format.numberFormat = "@";
  sheet.getRange("B2:V10802").format.numberFormat = "0.0000";
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
}

workbook.recalculate();
const outputPath = path.resolve(outputArg);
const previewDir = path.resolve(previewArg);
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
for (const [sheetName, stem] of [["温度", "temperature"], ["水分浓度", "moisture"]]) {
  for (const [range, suffix] of [["A1:V18", "top"], ["A10793:V10802", "bottom"]]) {
    const preview = await workbook.render({ sheetName, range, scale: 1.3, format: "png" });
    await fs.writeFile(path.join(previewDir, `${stem}_${suffix}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
for (const sheetName of ["温度", "水分浓度"]) {
  const top = await saved.inspect({
    kind: "table", sheetId: sheetName, range: "A1:V3", include: "values,formulas",
    tableMaxRows: 3, tableMaxCols: 22, maxChars: 7000,
  });
  const bottom = await saved.inspect({
    kind: "table", sheetId: sheetName, range: "A10800:V10802", include: "values,formulas",
    tableMaxRows: 3, tableMaxCols: 22, maxChars: 7000,
  });
  console.log(top.ndjson); console.log(bottom.ndjson);
}
const errors = await saved.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 }, summary: "result2 final formula error scan",
});
console.log(errors.ndjson);
console.log(`已导出：${outputPath}`);
