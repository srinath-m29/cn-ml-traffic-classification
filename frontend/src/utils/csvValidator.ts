/**
 * CSV Validation and Parser Engine
 * Validates uploaded network traffic flow CSV files against the 61 required model features.
 */

import Papa from 'papaparse';

export const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024; // 25 MB

export interface CsvValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  rowCount: number;
  parsedSamples: Record<string, number>[];
  headers: string[];
}

/**
 * Validates a CSV File or raw CSV text against the expected 61 model features.
 */
export async function parseAndValidateCsv(
  fileOrText: File | string,
  expectedFeatures: string[]
): Promise<CsvValidationResult> {
  // 1. File size check (if File object)
  if (typeof fileOrText !== 'string') {
    if (fileOrText.size > MAX_FILE_SIZE_BYTES) {
      return {
        isValid: false,
        errors: ['File is too large. Maximum supported size is 25 MB.'],
        warnings: [],
        rowCount: 0,
        parsedSamples: [],
        headers: [],
      };
    }
  }

  // 2. Read content
  const csvString = typeof fileOrText === 'string'
    ? fileOrText
    : await readFileAsText(fileOrText);

  if (!csvString || csvString.trim().length === 0) {
    return {
      isValid: false,
      errors: ['CSV file is empty.'],
      warnings: [],
      rowCount: 0,
      parsedSamples: [],
      headers: [],
    };
  }

  // 3. Parse with PapaParse
  const parseResult = Papa.parse<Record<string, string>>(csvString, {
    header: true,
    skipEmptyLines: 'greedy',
    transformHeader: (header: string) => header.trim(),
  });

  if (parseResult.errors && parseResult.errors.length > 0) {
    const parseErrors = parseResult.errors.slice(0, 5).map(
      (e) => `CSV Parse Error at row ${e.row ?? 'unknown'}: ${e.message}`
    );
    return {
      isValid: false,
      errors: parseErrors,
      warnings: [],
      rowCount: 0,
      parsedSamples: [],
      headers: [],
    };
  }

  const dataRows = parseResult.data;
  if (!dataRows || dataRows.length === 0) {
    return {
      isValid: false,
      errors: ['CSV file contains no data rows.'],
      warnings: [],
      rowCount: 0,
      parsedSamples: [],
      headers: [],
    };
  }

  const rawHeaders = parseResult.meta.fields || [];
  const headerSet = new Set(rawHeaders);
  const expectedSet = new Set(expectedFeatures);

  const errors: string[] = [];
  const warnings: string[] = [];

  // 4. Check for missing required features
  const missingFeatures: string[] = [];
  for (const feature of expectedFeatures) {
    if (!headerSet.has(feature)) {
      missingFeatures.push(feature);
    }
  }

  if (missingFeatures.length > 0) {
    if (missingFeatures.length <= 5) {
      for (const missing of missingFeatures) {
        errors.push(`Missing required feature: ${missing}`);
      }
    } else {
      errors.push(
        `Missing ${missingFeatures.length} required features (e.g. ${missingFeatures.slice(0, 3).join(', ')}...)`
      );
    }
  }

  // 5. Check for unexpected columns
  const unexpectedColumns: string[] = [];
  for (const col of rawHeaders) {
    if (!expectedSet.has(col)) {
      unexpectedColumns.push(col);
    }
  }

  if (unexpectedColumns.length > 0) {
    for (const extra of unexpectedColumns) {
      errors.push(`Unexpected column: "${extra}" (only the ${expectedFeatures.length} model features are expected)`);
    }
  }

  // If header check already failed, return early without validating individual rows
  if (errors.length > 0) {
    return {
      isValid: false,
      errors,
      warnings,
      rowCount: dataRows.length,
      parsedSamples: [],
      headers: rawHeaders,
    };
  }

  // 6. Validate row-level numeric values
  const parsedSamples: Record<string, number>[] = [];
  let rowErrorCount = 0;
  const maxRowErrorsToReport = 10;

  for (let rowIndex = 0; rowIndex < dataRows.length; rowIndex++) {
    const rawRow = dataRows[rowIndex];
    const cleanSample: Record<string, number> = {};
    let rowHasError = false;

    for (const feature of expectedFeatures) {
      const valStr = rawRow[feature];

      if (valStr === undefined || valStr === null || valStr.trim() === '') {
        rowErrorCount++;
        if (errors.length < maxRowErrorsToReport) {
          errors.push(`Empty or missing value for "${feature}" at row ${rowIndex + 1}`);
        }
        rowHasError = true;
        break;
      }

      const numVal = Number(valStr);
      if (isNaN(numVal) || !isFinite(numVal)) {
        rowErrorCount++;
        if (errors.length < maxRowErrorsToReport) {
          errors.push(`Invalid numeric value in "${feature}" at row ${rowIndex + 1}: received "${valStr}"`);
        }
        rowHasError = true;
        break;
      }

      cleanSample[feature] = numVal;
    }

    if (!rowHasError) {
      parsedSamples.push(cleanSample);
    }
  }

  if (rowErrorCount > maxRowErrorsToReport) {
    errors.push(`... and ${rowErrorCount - maxRowErrorsToReport} additional row data errors.`);
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    rowCount: dataRows.length,
    parsedSamples: errors.length === 0 ? parsedSamples : [],
    headers: rawHeaders,
  };
}

function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error('Failed to read file contents'));
    reader.readAsText(file);
  });
}
