Hi!

I'm Jesús Caballero, a Sauti Market Monitoring market team member, specialy from the DS team and 'll try to explain you were we got and what we think could be done from this point forward. 
As you may know, this project is meant to provide real-time prices of the commodities in the East of Africa, but more important than that, the phase of the prices.
If a price is on a particulary phase, the aid workers could sent the help to the specific community in need. 

We chose to rely in a industry standard index called AlPS, which was developed by the World Food Program. You may find their methodology in a PDF in this repository.
Due the raw data, that is poor in terms of lenght, we only could work with the ALPS for only four time series of almost 10K. So, we had to think another possible way.
The ALPS demands at least 3 years of historical data, but I felt we should considered 4 years or more of historical data, in order to capture better seasonalities.
On the other hand, our Stakeholder suggested us to relax the requirements. Consider less leght of data and see what happens. We did that. We relaxed to 2 years of data.
In the case, we named the metholody as weak ALPS.

Even relaxing the ALPS requirements we were getting arround 28 time series with their prices labeled. We need to think in another way. 
So we decided to change the base ALPS which is Linear Regression to an ARIMA (forecasting). This was a success for we, we get more than 100 time series with phases.



This is the list of notebooks you may found in this repository:

  - Global Methodology. 
    
    It explains the main methodology. 
