import { fa, fakerDE as faker } from "@faker-js/faker";

import crypto from "crypto";
import fs from "fs";
import Sql from "./Sql.js";
await Sql.start();
import { Parser } from "json2csv";

import { readFile } from "fs/promises";
const inf = JSON.parse(await readFile(new URL("./inf.json", import.meta.url)));

function getOccurrence(array, value) {
  var count = 0;
  array.forEach((v) => v === value && count++);
  return count;
}
function arrayToCSV(data) {
  let ans = "";
  let returnValue = "";
  const tmp_regex = /\\n/g;
  let dataF = data.map((el) => {
    return el.map((lx) => {
      returnValue += `${Object.values(lx).join(",")}\n`;
    });
  });
  ans = `${Object.keys(data[0][0]).join(",")}\n${returnValue}`;
  return ans.replaceAll(tmp_regex, "\n");

  let csv = data.map((row) => {
    returnValue += `${Object.values(row).join(",")}\n\r\r`;
  });
  ans = `${Object.keys(data[0]).join(",")}\n\r\r${returnValue}`;

  return ans.replaceAll(
    "\n\r\r",
    `
  `
  );
}
async function createRatingsMatrix() {
  const user = "00e6d9e9-f5bb-49a5-9e16-af3dd2ba9e3f";
  const numUsers = await Sql.$queryRaw(`SELECT * FROM user`, []);
  const numVideos = await Sql.$queryRaw(`SELECT * FROM video`, []);
  const reviews = await Sql.$queryRaw(`SELECT * FROM review`, []);
  const views = await Sql.$queryRaw(`SELECT * FROM view`, []);
  var ratingsMatrix = [];
  let i = true;

  // Initialize the matrix with zeros

  let userGroup = [];

  let userLocationGroup = [];
  let similarity = [];

  const videoGiven = numUsers.filter((el) => el.user_id === user)[0];
  let sum = 0;
  for (let i = 0; i < numUsers.length; i++) {
    userGroup = numUsers.filter((el) => {
      if (el.age > videoGiven.age) {
        return el.age - videoGiven.age < 2;
      }
      return videoGiven.age - el.age < 2;
    });
  }
  for (let i = 0; i < numUsers.length; i++) {
    userLocationGroup = numUsers.filter((el) => {
      return videoGiven.location === el.location;
    });
  }

  for (let mx = 0; mx < numUsers.length / 5; mx++) {
    similarity[mx] = [];
    ratingsMatrix[mx] = [];
    for (let j = 0; j < numVideos.length; j++) {
      let m = 0;
      let averageReview = 0;
      let videos = 0;

      for (let xi = 0; xi < userGroup.length; xi++) {
        if (numVideos[j].location === userGroup[xi]?.location) {
          m += 0.02;
        }
        if (
          numVideos[j].length > userGroup[xi].average_watch_time
            ? numVideos[j].length - userGroup[xi].average_watch_time < 2
            : userGroup[xi].average_watch_time - numVideos[j].length < 2 &&
              userGroup[xi].average_watch_time !== 0
        ) {
        }
        for (let z = 0; z < views.length; z++) {
          if (
            views[z].viewer === userGroup[xi].id &&
            views[z].video === numVideos[j].id
          ) {
            m += views[z].watch_time * 0.005;
            m += 0.025;
          }
        }
      }
      for (let zi = 0; zi < userLocationGroup.length; zi++) {
        if (numVideos[j].location === userLocationGroup[zi]?.location) {
          m += 0.02;
        }
        if (
          numVideos[j].length > userLocationGroup[zi].average_watch_time
            ? numVideos[j].length - userLocationGroup[zi].average_watch_time < 2
            : userLocationGroup[zi].average_watch_time - numVideos[j].length <
                2 && userLocationGroup[zi].average_watch_time !== 0
        ) {
          m += 0.125;
        }
        for (let z = 0; z < views.length; z++) {
          if (
            views[z].viewer === userLocationGroup[zi].id &&
            views[z].video === numVideos[j].id
          ) {
            m += views[z].watch_time * 0.005;
            m += 0.025;
          }
        }
      }

      if (
        numVideos[j].location ===
        numUsers.filter((ex) => ex.user_id === user)[0].location
      ) {
        similarity[mx].push({ location: numVideos[j].location });
        m += 2;
      }

      for (let k = 0; k < reviews.length; k++) {
        if (
          reviews[k].watcheduser === numUsers[mx].user_id &&
          reviews[k].watchedvideo === numVideos[j].video_id
        ) {
          videos += 1;
          sum += 1;
          m += 0.4;
        }
      }

      ratingsMatrix[mx][j] = {
        numberOfViews: videos,
      };
      m += (averageReview * 0.5) / (videos ? videos : 1);
      for (let z = 0; z < views.length; z++) {
        if (
          views[z].viewer ===
            numUsers.filter((ex) => ex.id === user)[0].user_id &&
          views[z].video === numVideos[j].id
        ) {
          m += views[z].watch_time * 0.01;
          m += 0.2;
        }
      }
      if (
        numVideos[j].length >
        numUsers.filter((ex) => ex.user_id === user)[0].average_watch_time
          ? numVideos[j].length -
              numUsers.filter((ex) => ex.user_id === user)[0]
                .average_watch_time <
            2
          : numUsers.filter((ex) => ex.user_id === user)[0].average_watch_time -
              numVideos[j].length <
              2 &&
            numUsers.filter((ex) => ex.user_id === user)[0]
              .average_watch_time !== 0
      ) {
        ratingsMatrix[mx][j] = {
          length: numVideos[j].length,
        };
        m += 1.25;
      }

      ratingsMatrix[mx][j] = {
        ...ratingsMatrix[mx][j],
        m: m,
        ...numVideos[j],
        ...numUsers[mx],
      };
    }

    // ratingsMatrix = ratingsMatrix
    //   .map((item, idx) => {
    //     return {
    //       item,
    //       idx,
    //     };
    //   })
    //   .sort((a, b) => b.item - a.item);

    // Fill the matrix with ratings from reviews data
    const filtered = ratingsMatrix.slice(0, 10);
    const items = [];

    // filtered.forEach((item) => items.push(numVideos[item.idx]));
  }
  return ratingsMatrix;
}
const datasa = await createRatingsMatrix();
console.log(arrayToCSV(datasa));
fs.writeFileSync(
  "./response.csv",
  JSON.stringify(arrayToCSV(await createRatingsMatrix())),
  () => {}
);
// const viewsFROMDB = await Sql.$queryRaw(`SELECT * FROM view`, []);

var mysql_data = JSON.parse(JSON.stringify(views));

//convert JSON to CSV Data

var file_header = Object.keys(views[0]);

var json_data = new Parser({ file_header });

var csv_data = json_data.parse(mysql_data);
fs.writeFileSync("./views.csv", csv_data, (e) => {});

const tf = require("@tensorflow/tfjs");
import csv from "csv-parser";

// Step 1: Preprocess the data
const videos = [];
const users = [];
const views = [];
const reviews = [];
const mergedData = [];

fs.createReadStream("video.csv")
  .pipe(csv())
  .on("data", (data) => {
    videos.push(data);
  })
  .on("end", () => {
    fs.createReadStream("users.csv")
      .pipe(csv())
      .on("data", (data) => {
        users.push(data);
      })
      .on("end", (data) => {
        // Processed user data
        fs.createReadStream("views.csv")
          .pipe(csv())
          .on("data", (data) => {
            views.push(data);
          })
          .on("end", () => {
            fs.createReadStream("reviews.csv")
              .pipe(csv())
              .on("data", (data) => {
                reviews.push(data);
              })
              .on("end", () => {
                // Processed review data
              });
            // Processed view data
          });
      });
    // Processed video data
  });
fs.writeFile("./response.csv", JSON.stringify(mergedData), (e) => {});
// Step 2: Prepare the data for training
// Merge the data from different CSV files and create a unified dataset

// Step 3: Train the recommendation model
// Choose a suitable model architecture and train it using TensorFlow.js

// Step 4: Evaluate the model
// Split the dataset into training and validation sets, and measure the model's performance

// Step 5: Generate recommendations
// Use the trained model to generate recommendations for users
// const mergedData = mergeData(videos, users, views, reviews);

// Convert the merged data into the desired format for training
const trainingData = convertToTrainingFormat(mergedData);

// Assuming the data arrays (videos, users, views, reviews) have been populated

// Loop through the views array to merge relevant data

console.log("");
let datas = [];
