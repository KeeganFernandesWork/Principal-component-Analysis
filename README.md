
### Explaining PCA

Principal Component Analysis(PCA) is an essential algorithm in a data scientist's toolkit. It is used to reduce the dimensionality of a dataset while retaining as much of the original information as possible. This makes it particularly useful for analyzing large datasets with many variables, where it can be difficult to visualize and interpret the data. We will understand its uses, working, and implementation. We will also discuss some common pitfalls and limitations of PCA and how to overcome them.

#### Why We Use PCA

One of the top uses of PCA is visualization. To show why, we need to see the various representations of data.

  

![](https://cdn-images-1.medium.com/max/800/0*DiZqBFhXxSZc-SUu.png)

The image above is one-dimensional data. We can represent it in a line and discriminate between apples, tomatoes and bananas.
```
import plotly.express as px  
df = px.data.iris() # iris is a pandas DataFrame  
fig = px.scatter(df, x="sepal_width", y="sepal_length",color ="species")  
fig.show()
```

![](https://cdn-images-1.medium.com/max/800/1*vN-1hxmhd1Fr6dOl9p9Ayg.png)

The image above is a 2D scatter plot representing the two variables sepal_length and sepal_width. We can see how we can discriminate between the various species using the two columns.

```
import plotly.express as px  
df = px.data.iris() # iris is a pandas DataFrame  
fig = px.scatter_3d(df, x="sepal_width", y="sepal_length",z='petal_width',color ="species")  
fig.show()
```

![](https://cdn-images-1.medium.com/max/800/1*yyL2dO88-cBbuH-FlDnrQw.png)

And finally, we can represent and classify the data using 3D plots. However, this is the hard limit of data visualisation since humans can’t see dimensions above four, so we can use Principal Component Analysis to pick the best components(1,2,3) to represent the data.

#### Mathematical Explanation of PCA

It’s important to understand the mathematics behind an Algorithm to understand the usability of the dataset. Unfortunately, this concept is complex and can’t be explained as well using the written word alone. I find the channel [StatQuest](https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw) is able to explain this concept quite well in the following video.

[PCA step-by-step](https://www.youtube.com/watch?v=FgakZw6K1QQ)

#### Implementing PCA with Python

We will use scikit-learn’s library to make a component analysis of our 4-dimensional data.

I will be using a stroke prediction dataset that can be found on [Aishwarya Ramakrishnan’s](https://github.com/aishwarya8615) gist repository. The full notebook can be found on my Github Page and my collab notebook.

#### Loading the dependencies we will be using  
```
import pandas as pd  
import matplotlib.pyplot as plt  
import plotly.express as px  
import seaborn as sns
```

# Loading the Dataset into memory  
```
url = "https://gist.githubusercontent.com/aishwarya8615/d2107f828d3f904839cbcb7eaa85bd04/raw/cec0340503d82d270821e03254993b6dede60afb/healthcare-dataset-stroke-data.csv"  
df = pd.read_csv(url,index_col = 0)  
df
```

After loading the Dataset, we will do some preprocessing and EDA, which can be found in my notebook and then proceed to Principal Component Analysis.
```
from sklearn.decomposition import PCA  
pca = PCA(n_components = 3)  
```
### These are the variabls that we will be using for the PCA  
```
cols = ["age", "avg_glucose_level","bmi","hypertension", "heart_disease"]  
pca.fit(df[["age", "avg_glucose_level","bmi","hypertension", "heart_disease"]])  
  
  
Array = pd.DataFrame(pca.fit_transform(df[["age", "avg_glucose_level","bmi","hypertension", "heart_disease"]]).tolist(),columns = ["PC1","PC2","PC3"])  
Array.info()  
  
Array["stroke"] = list(df["stroke"])  
  
px.scatter_3d(Array,x = "PC1" , y= "PC2" ,z = "PC3" ,color = "stroke")
```
![](https://cdn-images-1.medium.com/max/800/1*50AL9TVoiliAB-qpUYDmLg.png)

Although faint, one can clearly see a linear separation in the data at the 0 of the x-axis. This shows the data will likely be classified using linear algorithms.

#### Conclusion

Principal Component Analysis (PCA) is a powerful technique in data analysis and machine learning that can help us identify the underlying patterns in complex datasets. Mastering PCA can greatly enhance your ability to extract insights and make informed decisions from complex data, making it an essential skill for any data scientist or machine learning practitioner.
