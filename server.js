import crypto from "crypto";
import fs from "fs";
import { Parser } from "json2csv";
import csv from "csv-parser";
import { readFile } from "fs/promises";
import converter from "json-2-csv";
const inf = JSON.parse(await readFile(new URL("./inf.json", import.meta.url)));

const users = [];
fs.createReadStream("./polished.csv")
  .pipe(csv())
  .on("data", (data) => {
    users.push(data);
  })
  .on("end", (data) => {
    // Processed user data
    const uniqueObjects = Array.from(
      new Set(users.map((item) => item.name))
    ).map((name) => {
      return users.find((item) => item.name === name);
    });
    converter
      .json2csv(
        uniqueObjects.map((e) => {
          return {
            ...e,
            rating: crypto.randomInt(5),
          };
        }),
        { excelBOM: true, expandArrayObjects: true, unwindArrays: true }
      )
      .then((e) => {
        fs.writeFileSync("./movies_with_rating.csv", e);
        console.log("Complete");
      });
  });
