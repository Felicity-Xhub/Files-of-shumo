import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [templateArg, moistureArg, outputArg, previewArg] = process.argv.slice(2);
if (!templateArg || !moistureArg || !outputArg || !previewArg) {
  throw new Error("用法：node build_result3.mjs <模板.xlsx> <水分.csv> <输出.xlsx> <预览目录>");
}

async function readCsv(csvPath) {
  const text = (await fs.readFile(path.resolve(csvPath), "utf8")).replace(/^\uFEFF/, "").trim();
  const lines = text.split(/\r?\n/);
  if (lines.length < 2) throw new Error("水分CSV没有数据行。");
  return lines.map((line, rowIndex) => {
    const cells = line.split(",");
    if (cells.length !== 22) throw new Error(`${csvPath} 第${rowIndex + 1}行不是22列。`);
    return cells.map((cell, columnIndex) => {
      if (rowIndex === 0 && columnIndex === 0) return "时间\\到药材中心的距离/cm";
      if (rowIndex === 0) {
        const numeric = Number(cell);
        return Number.isInteger(numeric) ? String(numeric) : numeric.toFixed(1);
      }
      const numeric = Number(cell);
      if (!Number.isFinite(numeric)) throw new Error(`${csvPath} 存在非有限数值。`);
      // 保留求解器输出的未舍入数值，仅用Excel格式显示四位小数。
      return numeric;
    });
  });
}

const data = await readCsv(moistureArg);
const lastRow = data.length;
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path.resolve(templateArg)));
const sheet = workbook.worksheets.getItem("Sheet1");
sheet.getRange("A1:F5").clear({ applyTo: "contents" });
sheet.getRange("A1").writeValues(data);
const fullRange = sheet.getRange(`A1:V${lastRow}`);
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
sheet.getRange(`A1:A${lastRow}`).format.columnWidth = 27;
sheet.getRange(`B1:V${lastRow}`).format.columnWidth = 10;
sheet.getRange(`A2:A${lastRow}`).format.numberFormat = "0.0000";
sheet.getRange("B1:V1").format.numberFormat = "@";
sheet.getRange(`B2:V${lastRow}`).format.numberFormat = "0.0000";
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(1);

workbook.recalculate();
const outputPath = path.resolve(outputArg);
const previewDir = path.resolve(previewArg);
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
for (const [range, suffix] of [
  ["A1:V18", "top"],
  [`A${Math.max(2, lastRow - 9)}:V${lastRow}`, "bottom"],
]) {
  const preview = await workbook.render({ sheetName: "Sheet1", range, scale: 1.3, format: "png" });
  await fs.writeFile(path.join(previewDir, `${suffix}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
for (const range of ["A1:V3", `A${Math.max(2, lastRow - 2)}:V${lastRow}`]) {
  const check = await saved.inspect({
    kind: "table", sheetId: "Sheet1", range, include: "values,formulas",
    tableMaxRows: 3, tableMaxCols: 22, maxChars: 7000,
  });
  console.log(check.ndjson);
}
const errors = await saved.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 }, summary: "result3 final formula error scan",
});
console.log(errors.ndjson);
console.log(`已导出：${outputPath}`);
