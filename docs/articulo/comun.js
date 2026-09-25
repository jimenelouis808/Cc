// Piezas compartidas por el generador del documento.
const fs = require("fs");
const {
  Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, convertInchesToTwip,
} = require("docx");

const ACCENT = "1F3864";
const HEAD_BG = "E8EDF5";
const W = 9360;                  // ancho útil, carta con márgenes de 1"
const PX = 624;                  // 6.5 pulgadas a 96 ppp

function pngSize(ruta) {
  const b = fs.readFileSync(ruta);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function p(text, o = {}) {
  return new Paragraph({
    spacing: { after: o.after === undefined ? 120 : o.after, before: o.before || 0, line: 276 },
    alignment: o.align, indent: o.indent, border: o.border,
    children: [new TextRun({
      text, size: o.size || 21, font: "Calibri",
      bold: o.bold, italics: o.italics, color: o.color || "1A1A1A" })],
  });
}
function rich(runs, o = {}) {
  return new Paragraph({
    spacing: { after: o.after === undefined ? 120 : o.after, line: 276 },
    indent: o.indent,
    children: runs.map(r => new TextRun({
      text: r.t, size: r.size || 21, font: r.mono ? "Consolas" : "Calibri",
      bold: r.b, italics: r.i, color: r.c || "1A1A1A" })),
  });
}
function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160 },
    children: [new TextRun({ text, size: 30, bold: true, font: "Calibri", color: ACCENT })] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 110 },
    children: [new TextRun({ text, size: 24, bold: true, font: "Calibri", color: ACCENT })] });
}
function h3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3,
    spacing: { before: 220, after: 90 },
    children: [new TextRun({ text, size: 21, bold: true, font: "Calibri", color: "2E4B7A" })] });
}
function bullet(text, level = 0) {
  return new Paragraph({ numbering: { reference: "vinetas", level },
    spacing: { after: 80, line: 276 },
    children: [new TextRun({ text, size: 21, font: "Calibri", color: "1A1A1A" })] });
}
function code(text) {
  return new Paragraph({ spacing: { after: 60, before: 60 },
    indent: { left: convertInchesToTwip(0.3) },
    children: [new TextRun({ text, size: 18, font: "Consolas", color: "333333" })] });
}
function cita(text) {
  return new Paragraph({
    spacing: { after: 140, before: 80, line: 276 },
    indent: { left: convertInchesToTwip(0.32), right: convertInchesToTwip(0.2) },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 12 } },
    children: [new TextRun({ text, size: 20, font: "Calibri", italics: true, color: "1A1A1A" })] });
}
function figura(ruta, pie, ancho = PX) {
  const { w, h } = pngSize(ruta);
  const alto = Math.round(ancho * h / w);
  const out = [new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(ruta),
                              transformation: { width: ancho, height: alto } })] })];
  if (pie) out.push(new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: pie, size: 17, font: "Calibri", color: "4A4A4A" })] }));
  return out;
}
function table(headers, rows, widths) {
  const cw = widths || headers.map(() => Math.floor(W / headers.length));
  const mk = (text, i, isHead) => new TableCell({
    width: { size: cw[i], type: WidthType.DXA },
    shading: isHead ? { type: ShadingType.CLEAR, fill: HEAD_BG, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({ spacing: { after: 0, line: 240 },
      children: [new TextRun({ text, size: 17, font: "Calibri", bold: isHead,
                               color: isHead ? ACCENT : "1A1A1A" })] })] });
  return new Table({
    columnWidths: cw, width: { size: W, type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color: "BFBFBF" },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: "BFBFBF" },
      left: { style: BorderStyle.SINGLE, size: 2, color: "BFBFBF" },
      right: { style: BorderStyle.SINGLE, size: 2, color: "BFBFBF" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: "D9D9D9" },
      insideVertical: { style: BorderStyle.SINGLE, size: 1, color: "D9D9D9" } },
    rows: [new TableRow({ tableHeader: true, children: headers.map((t, i) => mk(t, i, true)) }),
           ...rows.map(r => new TableRow({ children: r.map((t, i) => mk(t, i, false)) }))] });
}
function spacer(h = 160) { return new Paragraph({ spacing: { after: h }, children: [] }); }

module.exports = { ACCENT, W, PX, p, rich, h1, h2, h3, bullet, code, cita,
                   figura, table, spacer, pngSize };
