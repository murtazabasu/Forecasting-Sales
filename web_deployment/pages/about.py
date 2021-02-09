import streamlit as st
from PIL import Image
import config



def write():
    """Used to write the page in the app.py"""
    with st.spinner("Loading About..."):  
        image_tomatoes = Image.open(config.MEDIA + 'tomatoes.jpg')
        
        # display image
        st.image(
            image_tomatoes,
            use_column_width=True
        )
        
        st.write(
            """
## Case Study: Forecasting Daily Sales of Tomatoes

A customer wants to forecast the sales of tomatoes on a daily basis (1-day-ahead), aggregated over approximately 100
stores in order to optimize supply chain planning. Currently, the customer uses a simple forecast, he uses the sales one week before on the same weekday as forecast for the
week to come (for e.g. forecast for coming Tuesday = sales last Tuesday)

We have developed a model using machine learning that takes different features from the data as
input and gives a more better forecast for the sales of the tomatoes for the next day.

### Feature definition for the given data:

1. **Date:** Date of the recorded observation (datetime).
2. **salesAmount:** Sales amount on that day (int).
3. **promotionYes:** Did Promotion took place on this day (1: True; 0: False).
4. **bridgeDayYes:** A day taken off from work to fill the gap between a holiday Thursday (or Tuesday) and the weekend is called a bridge day. (1: True; 0: False).
5. **publicHolidayYes:** Was it a a public holiday (1: True; 0: False).
6. **seasonCode:** Code for defining a season (int --> 1,2,3,4).
7. **seasonName:** Description of the season (str --> 1: Frühling, 2: Sommer; 3: Herbst, 4: Winter).
8. **Calender Week:** Calender week for each year (int).
9. **newSnowCode:** Code for defining the description of the snow (int --> 0, 1, 2). 
10. **newSnowDescription:** Description of the snow code (str --> 0: kein Schnee; 1: mehr als 1 cm Schnee; 2: weniger als 1 cm Schnee).
11. **rainCode:** Code for defining the description of the rain (int --> 1, 2, 3, 4, 5).
12. **rainDescription:** Description of the rain code (str --> 1: viel trockener als mittel; 2: trockener als mittel; 3: Mittel, 4: feuchter als mittel, 5: viel feuchter als mittel).  
13. **sunshineCode:** Code for defining the description of the sunshine (int --> 1, 2, 3, 4, 5). 
14. **sunShineDescription:** Description of the sunshine code (str --> 1: viel weniger als mittel; 2: weniger als mittel; 3: Mittel, 4: mehr als mittel, 5: viel mehr als mittel).
15. **shopsClosedYes:** Is the shop closed? (1: True, 0: False).  
16. **weekdayCode:** Code for each day in the week. (int --> 1, 2, 3, 4, 5, 6, 7).  
17. **weekDayDescription:** Name of the day. (str --> 1: Sonntag; 2: Montag; 3: Dienstag; 4: Mittwoch; 5: Donnerstag; 6: Freitag; 7: Samstag).        
       """
        )