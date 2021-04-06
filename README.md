# emotion-recognition

<h1>About the project 💻</h1>

The goal of the project was to design and develop software for facial emotion recognition based on real time images. The whole design was based on the solution proposed in the article <i>"An Efficient Automatic Facial Expression Recognition using Local Neighborhood Feature Fusion"</i>, by P. Shanthi, S. Nickolas.

For an efficient texture representation, there are two descriptors combined, Local Binary Patern (LBP) and Local Neighborhood Encoded Pattern (LNEP). 

SVM model has a 93,83% accuracy. Unfortunately, when applied in real time accuracy is much worse.

<h2>Used libraries 📚</h2>

<ul>
  <li>matplotlib</li>
  <li>numpy</li>
  <li>openCV</li>
  <li>Scikit-learn</li>
</ul>

<h2>Database 👧</h2>

I used the extended Cohn-Kanade (CK+) database, which contains images of over 97 individuals displaying basic emotions: anger, happiness, contempt, sadness, surprise, fear, disgust.

<h2>Software stages ✔️</h2>

<ol>
  <li>Image preprocessing</li>
  <li>Using LBP and LNEP descriptors to create histograms with image features</li>
  <li>Combining the two histograms into one</li>
  <li>Dividing data into test and training sets</li>
  <li>Multi-class SVM classification</li>
  <li>Classification of the identified patterns</li>
</ol>

