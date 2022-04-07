Notes for Conceptualization

https://www.investopedia.com/articles/personal-finance/111115/zillow-estimates-not-accurate-you-think.asp
"As per the above article, there are definitely possibilities for data quality issues. Zillow had been relying on both the data shared by the users and publicly available datasets. It would be too harsh to say the data quality is garbage in the case of Zillow. Especially because their model is doing good in a number of cases. But here the impact of the error is huge. Let me explain this with an example,
Zillow is looking to buy a property ‘ABC’, then plans to invest some money in improvements and sell it in a few weeks for a profit. Let’s say Zillow’s Zestimate evaluates the property ‘ABC’ at $500,000. If the model had a 10% error, it means that the actual price should have been $450,000. Now, Zillow has bought the property at a $50,000 premium. It will be very difficult for Zillow to make a profit on this property."

Lesson: In any data science problem, there should be a lot of focus on the quality of the data. It will be good to have metrics in place to measure the quality of data. In scenarios like Zillow where a small error could result in a big error, there should be a way to at least partially verify 

https://towardsdatascience.com/in-defense-of-zillows-besieged-data-scientists-e4c4f1cece3c

"All of this is wrong. Zillow Offers and iBuying is an intrinsically risky business that was recklessly executed by Zillow’s business leaders who wanted to take over the real estate market in just a few years. As evidenced by Zillow’s competitors, Opendoor and Offerpad, iBuying does work! But it needs patience, experience, and a healthy respect for risk. Instead Zillow’s business leaders dove head-first into the shallow-end."

 An overestimate could wipe out profits, or worse, result in >$100M losses. An underestimate would result in rejected lowball offers and cripple Zillow’s ability to purchase homes (ie. low buyer conversion rates). While machine learning-based predictive models are standard practice in tech, housing is also subjected to significant and unpredictable macroeconomic trends.
 
 https://www.investopedia.com/articles/personal-finance/111115/zillow-estimates-not-accurate-you-think.asp
 
 Zestimates are only as accurate as the data behind them, so if the number of bedrooms or bathrooms in a home, its square footage, or its lot size are inaccurate on Zillow, the Zestimate will be off.
 
 The Zestimate also takes into account actual property taxes paid, exceptions to tax assessments, and other publicly available property tax data. Tax assessor’s property values can be inaccurate, though. The tax assessor’s database might have a mistake related to a property’s basic information, causing the assessed value to be too high or too low.
 
 The Bottom Line
Zillow isn’t trying to hide the imperfections of its Zestimates from consumers, and you can’t expect perfectly accurate estimates from competing sites, either. The point is for homeowners to use prices from Zestimates as a broad guideline, and contrast these figures against other sources. It should not, in any way, be considered an appraisal. A comparable market analysis from a local real estate agent and a professional appraisal of the home are the best ways to learn its value.


CHECKOUT LEAFMAP IF I GET THE TIME 
https://leafmap.org/notebooks/00_key_features/

"Although it represents location, like zipcode, in this circumstance a higher longitudinal or latitudinal value does represent a farther distance." Treat it as continuous. If it was extremely vague like just 38 N 118 W then it could MAYBE be categorical, and definitely discrete. 