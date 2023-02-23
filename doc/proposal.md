# **CMPUT 466 Project proposal**

## **Rocket League Rotation Analysis**

### **Description**

To understand this project, Rocket League is a team game about playing soccer with cars, competitive Rocket League consists of three players per team and one ball. Players drive around the pitch and try to hit the ball into the net much like normal soccer. There are many aspects such as boost collection, demolitions, bumping, flying through the air etc.

The primary objective is to predict and demonstrate positions and rotation directions for a team of players given the positions of the opposing team and the ball. The goal is to benefit the game knowledge of the players to improve overall skill.

The plan is to take a large sample of RLCS (Rocket League Championship Series) game replays, gather the data for each player and then create a machine learning model for it. Because RLCS is the highest competitive level of rocket league the learned model should be trained at a professional level predicting their rotations and positions. Inside these replays, there is vector data for each player and ball including their position, direciton, and speed. Of course no rotation is down to a science and there is lots of randomness and stochasticity.

This is a very simple subset of data that I plan to use, a much more complicated model would be taking into account boost levels, whether or not a boost pad is available on the field and playing for demolitions. If the first plan is completed with accuracy and robustness this will be the extension to this project.

I have searched the internet for anything relating to this, and it seems that this particular task has not been done, which makes for an interesting project. Everything will have to be original and thoroughly planned because there will be no reference material. However, there have been models employed to predict scoring chance, win chance during the game and others, so learning how to scrape replay data may not be the most difficult task.

### **Available Software**

Some software that I plan on using would be the python libraries PyTorch or TensorFlow. These Machine Learning libraries will assist with learning an appropriate model and applying it. I will be training this model on either a cloud based server or on my home pc which uses an Nvidia 1080ti 11Gb graphics card. Each replay file contains a large amount of data so the library Pandas may be used to read that data in. I will also be using a website called ballchasing.org to gather replays for training and testing. There is also a python library called carball that I may or may not use for replay data scraping.

### **Class to Class Topics**

I will use class topics such as MLE, gradient descent, and even perhaps looking forward some neural networks. Once I have 3 separate models using different methods, I will compare them and determine the best for this project.

### **Inspiration**

Inspiration for this project comes from my love for Rocket League and playing at a very high competetive level. I have also played team sports such as Hockey and Lacrosse my entire life and playing as a team with proper positions is a very important aspect. Being able to use a machine learning model to help me and my teammates improve would be amazing given that we dont have 24/7 access to the top coaches in the sport. 

### **Project Timeline**

- March 4: Complete the replay data scraping algorithm and solution for storing it
- March 11: Narrow down the inputs to simplify the model for a first round accomplishment
- March 18: Have a basic model learned and working with test data, do an analysis on the model to see what to improve
- March 25: Add additional features to the model to increase complexity and overall applicability
- April 1: Have a final working model complete and trained, report done and handed in.

Note: Each time a model is complete and trained, it will be complete and trained using 3 different methods.

### **Sources**

https://ballchasing.com/doc/api  

https://github.com/SaltieRL/carball  

https://alpscode.com/blog/rocket-league-and-sports-analytics/  

https://joeydotcomputer.substack.com/p/neuralnextg-v010-analyzing-rocket