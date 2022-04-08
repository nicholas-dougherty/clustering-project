Notes for Conceptualization

https://www.investopedia.com/articles/personal-finance/111115/zillow-estimates-not-accurate-you-think.asp
"As per the above article, there are definitely possibilities for data quality issues. Zillow had been relying on both the data shared by the users and publicly available datasets. It would be too harsh to say the data quality is garbage in the case of Zillow. Especially because their model is doing good in a number of cases. But here the impact of the error is huge. Let me explain this with an example,
Zillow is looking to buy a property ‘ABC’, then plans to invest some money in improvements and sell it in a few weeks for a profit. Let’s say Zillow’s Zestimate evaluates the property ‘ABC’ at $500,000. If the model had a 10% error, it means that the actual price should have been $450,000. Now, Zillow has bought the property at a $50,000 premium. It will be very difficult for Zillow to make a profit on this property."

Lesson: In any data science problem, there should be a lot of focus on the quality of the data. It will be good to have metrics in place to measure the quality of data. In scenarios like Zillow where a small error could result in a big error, there should be a way to at least partially verify 

*** 

https://towardsdatascience.com/in-defense-of-zillows-besieged-data-scientists-e4c4f1cece3c

"All of this is wrong. Zillow Offers and iBuying is an intrinsically risky business that was recklessly executed by Zillow’s business leaders who wanted to take over the real estate market in just a few years. As evidenced by Zillow’s competitors, Opendoor and Offerpad, iBuying does work! But it needs patience, experience, and a healthy respect for risk. Instead Zillow’s business leaders dove head-first into the shallow-end."

 An overestimate could wipe out profits, or worse, result in >$100M losses. An underestimate would result in rejected lowball offers and cripple Zillow’s ability to purchase homes (ie. low buyer conversion rates). While machine learning-based predictive models are standard practice in tech, housing is also subjected to significant and unpredictable macroeconomic trends.
***
 https://www.investopedia.com/articles/personal-finance/111115/zillow-estimates-not-accurate-you-think.asp
 
 Zestimates are only as accurate as the data behind them, so if the number of bedrooms or bathrooms in a home, its square footage, or its lot size are inaccurate on Zillow, the Zestimate will be off.
 
 The Zestimate also takes into account actual property taxes paid, exceptions to tax assessments, and other publicly available property tax data. Tax assessor’s property values can be inaccurate, though. The tax assessor’s database might have a mistake related to a property’s basic information, causing the assessed value to be too high or too low.
 
 The Bottom Line
Zillow isn’t trying to hide the imperfections of its Zestimates from consumers, and you can’t expect perfectly accurate estimates from competing sites, either. The point is for homeowners to use prices from Zestimates as a broad guideline, and contrast these figures against other sources. It should not, in any way, be considered an appraisal. A comparable market analysis from a local real estate agent and a professional appraisal of the home are the best ways to learn its value.
****

CHECKOUT LEAFMAP IF I GET THE TIME 
https://leafmap.org/notebooks/00_key_features/

"Although it represents location, like zipcode, in this circumstance a higher longitudinal or latitudinal value does represent a farther distance." Treat it as continuous. If it was extremely vague like just 38 N 118 W then it could MAYBE be categorical, and definitely discrete. 
***

https://vinvashishta.substack.com/p/zillow-just-gave-us-a-look-at-machine?s=r

The CEO said repeatedly, there was a narrow window of accuracy that would create a sustainable business. Everything else would result in failure. This is the connection between model metrics and business metrics. Small inaccuracy ranges could cause cascading failure.

  Too low on the initial home value estimations and homeowners would not accept the offer.

·         Too high on the initial estimation, margins would not be optimized.

·         Miss on the customer behavioral model, and homeowners would not accept the offers at a high enough rate.

·         Too low on the 6 month home value prediction and opportunities would be missed.

·         Too high on the 6 month estimate and margins would not be optimized or offers would be extended when they shouldn’t be.

The simulations and tests Zillow ran were incomplete. The assumption of stability is baked into most statistical models. In real world complex systems involving people making decisions, stability is NEVER an assumption that will hold in the long run. As I said in a previous post about experiments, these types of models will pass early validation, even using statistical experimental methods, then fail suddenly and catastrophically.

The problem is in the dynamics of real world data. The distribution changes. Training data’s distributions have distributions. The result is a probability of a starting distribution being part of serving accurate inference. I think of this in terms of inference spaces. When I advise Data Scientists to understand Topology and Differential Geometry, this is why.

The tooling and automation required to execute is a large scale project in and of itself. There is a massive tools gap here and custom built is the only way to go. Models need to be built to recommend data to be gathered and hypothesis to test. The distance between intervention and outcome makes it difficult for us to design experiments. Sifting through the data necessary to guess at those relationships is not something people can manually do in business timelines.
***
Creative Way

Partition your data according to target. You will end up with several disjoint subsets of data. Then start a statistical analysis on the association of variables within and between subsets of data. For instance the distribution of values within a variable should be significantly different among different classes, if that variable really contributes to that target (ANOVA).

This also helps to remove variables which do not contribute to the target (by doing this 1. you reduce the complexity of data and analysis and improve interpretability by removing them, 2. you already found which target IS NOT contributing which is a part of your answer)
***
#### Don't scale lat/long. 
If you have attributes with a well-defined meaning. Say, latitude and longitude, then you should not scale your data, because this will cause distortion. (K-means might be a bad choice, too - you need something that can handle lat/lon naturally)

If you have mixed numerical data, where each attribute is something entirely different (say, shoe size and weight), has different units attached (lb, tons, m, kg ...) then these values aren't really comparable anyway; z-standardizing them is a best-practise to give equal weight to them.

If you have binary values, discrete attributes or categorial attributes, stay away from k-means. K-means needs to compute means, and the mean value is not meaningful on this kind of data.
***
The aim of clustering would be to figure out commonalities and designs from the large data sets by splitting the data into groups. Since it is assumed that the data sets are unlabeled, clustering is frequently regarded as the most valuable unsupervised learning problem (Cios et al., 2007).

A primary application of geometrical measures (distances) to features having large ranges will implicitly assign greater efforts in the metrics compared to the application with features having smaller ranges. Furthermore, the features need to be dimensionless since the numerical values of the ranges of dimensional features rely upon the units of measurements and, hence, a selection of the units of measurements may significantly alter the outcomes of clustering. Therefore, one should not employ distance measures like the Euclidean distance without having normalization of the data sets (Aksoy and Haralick, 2001; Larose, 2005).


Preprocessing Luai et al. (2006) is actually essential before using any data exploration algorithms to enhance the results’ performance. Normalization of the dataset is among the preprocessing processes in data exploration, in which the attribute data are scaled tofall in a small specifiedrange. Normalization before clustering is specifically needed for distance metric, like the Euclidian distance that are sensitive to variations within the magnitude or scales from the attributes. In actual applications, due to the variations in selection of the attribute's value, one attribute might overpower another one. Normalization prevents outweighing features having a large number over features with smaller numbers. The aim would be to equalize the dimensions or magnitude and also the variability of those features.
***

https://mindmatters.ai/2021/12/zillows-house-flipping-misadventure/

#### However, it is often better to have good data than more data. A timeless aphorism is that the three most important things in real estate are location, location, location. A second, related aphorism is that all real estate is local. Data on homes thousands, hundreds, or even a few miles away are not necessarily relevant and possibly misleading. Even homes a few hundred feet apart can sell for quite different prices because of proximity to various amenities and disamenities, such as schools, parks, metro stations, noise pollution, and overhead power lines.

##### Even seemingly identical adjacent homes can sell for different prices because buyers and realtors, but not algorithms, might know, for example, that one house had been renovated recently or that someone had been murdered in one house.
##### I’ve seen homes sell for 50 to 100 percent over Zillow’s estimated market value because they were designed by a famous architect or owned by a celebrity. After the sale, Zillow increased the estimated market values of nearby homes because its algorithms did not know that the nearby homes had not been designed by famous architects or owned by celebrities.

Zillow’s original business model relied on advertising revenue from realtors, builders, and lenders but it has attempted to capitalize on its data and brand name by expanding into a variety of real-estate-related ventures, including a house-flipping business called Zillow Offers that was launched in 2018. 

However, informed sellers (who are also likely to have talked to local realtors) need not accept algorithmic generated offers. If the algorithmic offer is too high, the seller may well accept. If the algorithmic offer is too low, the seller is more likely to use a local realtor. On average, the algorithm will pay too much for the homes they buy.

Hypothesis
If parcel tax value is wide spread across the zip code, it should be more dificult for the ‘Zestimate’ to be accurate therefore abs(logerror) should be high .

****
Why Combine PCA and K-means Clustering?

There are varying reasons for using a dimensionality reduction step such as PCA prior to data segmentation. Chief among them? By reducing the number of features, we’re improving the performance of our algorithm. On top of that, by decreasing the number of features the noise is also reduced.

2. Data Preprocessing

Our segmentation model will be based on similarities and differences between individuals on the features that characterize them.

We’ll quantify these similarities and differences.

Well, you can imagine that two persons may differ in terms of ‘Age’. One may be a 20-year-old, while another – 70 years old. The difference in age is 50 years. However, it spans almost the entire range of possible ages in our dataset.

At the same time, the first individual may have an annual income of $100,000; while the second may have an annual income of $150,000. Therefore, the difference between their incomes will be $50,000.

Why?

Because the model is not familiar with our context. So, it defines one as age, while the other as income.

Therefore, it will place a much bigger weight on the income variable.

It is obvious that we must protect ourselves from such an outcome. What’s more, in general, we want to treat all the features equally. And we can achieve that by transforming the features in a way that makes their values fall within the same numerical range. Thus, the differences between their values will be comparable. This process is commonly referred to as standardization.

### Reasons for using PCA
https://365datascience.com/tutorials/python-tutorials/pca-k-means/