# Importing the necessary libraries

import pandas as pd
import numpy as np
import os
import datetime
import json
from os import listdir
from os.path import isfile,join
import datetime
import requests
import re

class DataCollection():

    if __name__ == '__main__':
        print('Collecting Data')
        DataCollection()

    def weekend(self,x):
        if x == 'Saturday' or x == 'Sunday':
            return 'Weekend'
        else:
            return 'Weekday'

    def get_mobility_data(self,rolling_mean=False):
        mob_index_url = 'https://raw.githubusercontent.com/descarteslabs/DL-COVID-19/master/DL-us-mobility-daterow.csv'

        df_mobility_index = pd.read_csv(mob_index_url, parse_dates=['date'])
        df_mobility_index = df_mobility_index.dropna(subset=['fips', 'admin2'])
        df_mobility_index.fips = df_mobility_index.fips.astype(int)
        df_mobility_index.fips = df_mobility_index.fips.astype(str).str.zfill(5)
        df_mobility_index = df_mobility_index.rename(columns={'admin1': 'STATE', 'admin2': 'COUNTY'})
        df_mobility_index = df_mobility_index.drop(columns=['country_code', 'admin_level'])
        df_mobility_index['weekday'] = df_mobility_index.date.dt.day_name()
        df_mobility_index = df_mobility_index[df_mobility_index.m50 < 200]
        df_mobility_index['weekend'] = df_mobility_index['weekday'].apply(lambda x: self.weekend(x))

        if rolling_mean:
            mob_col = ['m50']
            m50 = df_mobility_index.groupby(['fips', 'date'])[mob_col].mean()
            m50['rolling_mean_mob'] = m50[mob_col].rolling(7, min_periods=1).mean()
            m50 = m50.reset_index()
            return m50

        return df_mobility_index

    def get_spend_data(self):
        county_spending_url = 'https://raw.githubusercontent.com/OpportunityInsights/EconomicTracker/main/data/Affinity%20-%20County%20-%20Daily.csv'

        df_county_spending = pd.read_csv(county_spending_url, low_memory=False)

        df_county_spending['date'] = df_county_spending['year'].astype(str) + '-' + \
                                     df_county_spending['month'].astype(str) + '-' + \
                                     df_county_spending['day'].astype(str)
        df_county_spending.date = pd.to_datetime(df_county_spending.date, infer_datetime_format=True)
        df_county_spending.countyfips = df_county_spending.countyfips.astype(int)
        df_county_spending.countyfips = df_county_spending.countyfips.astype(str).str.zfill(5)
        df_county_spending = df_county_spending.rename(columns={'countyfips': 'fips'})
        df_county_spending.spend_all = df_county_spending.spend_all.str.replace('.', '0.')
        df_county_spending.spend_all = df_county_spending.spend_all.astype(float)
        return df_county_spending

    def get_covid_data(self,info='cases'):
        if info == 'cases':
            url = ('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'
                   'csse_covid_19_time_series/time_series_covid19_confirmed_US.csv').format(sep=2 * '\n')
            ren_col = 'Total_cases'
            val_col = 'Daily_cases'
        if info == 'deaths':
            url = ('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'
                   'csse_covid_19_time_series/time_series_covid19_deaths_US.csv').format(sep=2 * '\n')
            ren_col = 'Total_deaths'
            val_col = 'Daily_deaths'
        df = pd.read_csv(url)
        df = df[df['iso3'] == 'USA']
        df = df.dropna(subset=['FIPS'])
        df = df[df.FIPS <= 56045]

        df.FIPS = df.FIPS.astype(int).astype(str).str.zfill(5)
        df = df[(df.Admin2 != 'Unassigned') & (df.Admin2.notna())]
        df = df.drop(columns=['UID', 'iso2', 'iso3', 'code3', 'Country_Region', 'Combined_Key'])
        date_cols = [col for col in df.columns if '/20' in col]
        id_cols = [col for col in df.columns if col not in date_cols]
        df_gr = df.melt(id_vars=id_cols, value_vars=date_cols)
        df_gr[val_col] = df_gr.groupby(['FIPS'])[['value']].diff().fillna(0)
        df_gr = df_gr.rename(columns={'value': ren_col, 'variable': 'date',
                                      'Long_': 'Lng', 'Admin2': 'COUNTY',
                                      'Province_State': 'STATE', 'FIPS': 'fips'})
        df_gr.date = pd.to_datetime(df_gr.date, infer_datetime_format=True)
        df_gr = df_gr.drop(columns=['STATE', 'COUNTY'])
        return df_gr

    def get_covid_rates(self):
        url = 'https://raw.githubusercontent.com/OpportunityInsights/EconomicTracker/main/data/COVID%20-%20County%20-%20Daily.csv'
        df_covid_rate = pd.read_csv(url, low_memory=False)
        rate_cols = [col for col in df_covid_rate.columns if 'rate' in col]
        df_covid_rate['date'] = df_covid_rate['year'].astype(str) + '-' + \
                                df_covid_rate['month'].astype(str) + '-' + \
                                df_covid_rate['day'].astype(str)
        df_covid_rate['date'] = pd.to_datetime(df_covid_rate['date'], infer_datetime_format=True)
        all_cols = [col for col in df_covid_rate.columns if 'rate' in col or 'fips' in col or 'date' in col]

        df_covid_rate = df_covid_rate[all_cols]
        for col in rate_cols:
            df_covid_rate[col] = pd.to_numeric(df_covid_rate[col], errors='coerce')
            df_covid_rate[col] = df_covid_rate[col].fillna(0)

        df_covid_rate = df_covid_rate.rename(columns={'countyfips': 'fips'})
        df_covid_rate.fips = df_covid_rate.fips.astype(str).str.zfill(5)
        return df_covid_rate

    def combine_covid_data(self):
        df_covid_cases = self.get_covid_data(info='cases')
        df_covid_deaths = self.get_covid_data(info='deaths')[['fips', 'date', 'Total_deaths', 'Daily_deaths']]
        df_covid_merged = pd.merge(left=df_covid_cases, right=df_covid_deaths, on=['fips', 'date'])
        df_covid_rate = self.get_covid_rates()
        df_covid = pd.merge(left=df_covid_merged, right=df_covid_rate, on=['fips', 'date'], how='outer')
        df_covid = df_covid.dropna(subset=['Lat'])
        return df_covid

    def get_weather_chunk(self,lat, lng, start_dt, end_dt, API_KEY):

        url = 'https://api.worldweatheronline.com/premium/v1/past-weather.ashx'
        query = '?q=' + str(lat) + ',' + str(lng)
        key = '&key=' + API_KEY
        ret_format = '&format=json'
        sdate = '&enddate=' + start_dt
        edate = '&date=' + end_dt
        tp = '&tp=24'
        final_url = url + query + ret_format + edate + sdate + tp + key
        get_weather = requests.get(final_url)
        weather_data = json.loads(get_weather.text)
        return weather_data

    def get_weather_data(self,lat, lng, start_date, end_date, fips, key_count):
        new_low = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        new_end = datetime.timedelta(days=34) + new_low
        final_end = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        df = pd.DataFrame()
        with open('./config/keys.json') as outfile:
            keys = json.load(outfile)
        API_KEYS = keys['weather_api_keys']
        last = False
        API_KEY = API_KEYS[key_count]
        if final_end - new_low < datetime.timedelta(days=34):
            weather_data = self.get_weather_chunk(lat, lng, start_date, end_date, API_KEY)
            while ('error' in weather_data['data']):
                key_count += 1
                if key_count == len(API_KEYS):
                    raise Exception('Out of API Keys')
                API_KEY = API_KEYS[key_count]

                weather_data = self.get_weather_chunk(lat, lng, start_date, end_date, API_KEY)
            df = pd.DataFrame(weather_data['data']['weather'])
            last = True
        while (last == False):
            if new_end < final_end:

                end_dt = datetime.datetime.strftime(new_low, '%Y-%m-%d')
                start_dt = datetime.datetime.strftime(new_end, '%Y-%m-%d')

                weather_data = self.get_weather_chunk(lat, lng, start_dt, end_dt, API_KEY)
                while ('error' in weather_data['data']):
                    print(API_KEY)
                    key_count += 1
                    if key_count == len(API_KEYS):
                        raise Exception('Out of API Keys')
                    API_KEY = API_KEYS[key_count]
                    weather_data = self.get_weather_chunk(lat, lng, start_dt, end_dt, API_KEY)
                df_weather_chunk = pd.DataFrame(weather_data['data']['weather'])
                df = pd.concat([df, df_weather_chunk], ignore_index=True)
                new_low = new_end + datetime.timedelta(days=1)
                last_itr = new_end
                new_end = datetime.timedelta(days=34) + new_low


            else:
                new_low = last_itr + datetime.timedelta(days=1)
                new_end = final_end
                end_dt = datetime.datetime.strftime(new_low, '%Y-%m-%d')
                start_dt = datetime.datetime.strftime(new_end, '%Y-%m-%d')
                weather_data = self.get_weather_chunk(lat, lng, start_dt, end_dt, API_KEY)
                df_weather_chunk = pd.DataFrame(weather_data['data']['weather'])
                df = pd.concat([df, df_weather_chunk], ignore_index=True)
                last = True

        df = df.drop(columns=['uvIndex'])
        df_h = pd.DataFrame()

        for val in df['hourly'].values:
            df_h = pd.concat([df_h, pd.DataFrame(val)], ignore_index=True)

        df = pd.concat([df, df_h], axis=1)
        use_cols = ['date', 'tempC', 'maxtempC', 'mintempC', 'WindChillC', 'FeelsLikeC', 'visibilityMiles',
                    'HeatIndexC',
                    'avgtempC', 'windspeedMiles', 'winddirDegree', 'pressure', 'WindGustMiles', 'precipMM',
                    'totalSnow_cm', 'sunHour', 'DewPointC', 'humidity', 'uvIndex']
        df = df[use_cols]
        cols = [col for col in df.columns if col != 'date']
        df = df.apply(pd.to_numeric, errors='ignore')
        df.date = pd.to_datetime(df.date, infer_datetime_format=True)
        df['fips'] = fips
        return df, key_count

    def get_county_weather(self,df_mobility,df_covid):
        i= 0
        print('Getting Weather Data')
        start_date = datetime.datetime.strftime(df_mobility.date.max(), '%Y-%m-%d')
        end_date = datetime.datetime.strftime(df_mobility.date.min(), '%Y-%m-%d')
        fips_lat_lng = df_covid[['fips', 'Lat', 'Lng']].drop_duplicates().dropna().set_index('fips').to_dict('index')
        fpath1 = './Data/Weather_Files/'
        os.makedirs(fpath1, exist_ok=True)
        files_all = [fpath1 + f for f in listdir(fpath1) if isfile(join(fpath1, f))]
        weather_files = [f for f in files_all if 'Weather' in f]
        fips_done = [file.split('_')[3].split('.csv')[0] for file in weather_files]
        key_count = 0
        for fips, lat_lng in list(fips_lat_lng.items()):
            if fips in fips_done:
                weather_df_fips = pd.read_csv(fpath1 + 'Weather_FIPS_' + str(fips) + '.csv', parse_dates=['date'])
                if weather_df_fips.date.max() >= df_mobility.date.max():
                    if i ==0:
                        print('Weather Data seems to be updated')
                        i+=1
                    continue
                else:
                    if i ==0:
                        print('Updating Weather Data, It may take 15-20 minutes')
                        i+=1
                    end_date = datetime.datetime.strftime(weather_df_fips.date.max() + datetime.timedelta(days=1),
                                                          format='%Y-%m-%d')
                    start_date = datetime.datetime.strftime(df_mobility.date.max(), format='%Y-%m-%d')
            else:
                weather_df_fips = pd.DataFrame()
            lat, lng = lat_lng['Lat'], lat_lng['Lng']
            weather_df, key_count = self.get_weather_data(lat, lng, start_date, end_date, fips, key_count)
            weather_df = pd.concat([weather_df_fips, weather_df], ignore_index=True, axis=0)
            weather_df.to_csv(fpath1 + 'Weather_FIPS_' + str(fips) + '.csv', index=False)

    def combine_weather(self,df_mobility,df_covid):
        start_date = datetime.datetime.strftime(df_mobility.date.max(), '%Y-%m-%d')
        fpath = './Data/All_County_Weather_' + start_date + '.csv'
        if os.path.exists(fpath):
            df_weather = pd.read_csv(fpath, parse_dates=['date'])
        else:

            self.get_county_weather(df_mobility, df_covid)
            fpath1 = './Data/Weather_Files/'
            files_all = [fpath1 + f for f in listdir(fpath1) if isfile(join(fpath1, f))]
            weather_files = [f for f in files_all if 'Weather' in f]

            df_weather = pd.DataFrame()
            i = 0
            for file in weather_files:
                df = pd.read_csv(file)
                df_weather = pd.concat([df_weather, df], ignore_index=True)
                i += 1
                if i % 500 == 0:
                    print('Finised Records = ', i)
            df_weather.date = pd.to_datetime(df_weather.date, infer_datetime_format=True)
            df_weather.to_csv(fpath, index=False)
        df_weather.fips = df_weather.fips.astype(str).str.zfill(5)
        return df_weather
    def get_county_demo(self):
        df_county = pd.read_csv('./Data/nhgis0010_ds239_20185_2018_county.csv', encoding='latin')
        df_county['fips'] = df_county.STATEA.astype(str).str.zfill(2) + df_county.COUNTYA.astype(str).str.zfill(3)
        df_gini_income = pd.read_csv('./Data/nhgis0010_ds240_20185_2018_county.csv',
                                     encoding='latin')
        df_gini_income = df_gini_income[['STATE', 'COUNTY', 'AKGVE001']]
        df_county_demo = pd.merge(left=df_county, right=df_gini_income, on=['STATE', 'COUNTY'])
        cols_A = df_county_demo.columns[df_county_demo.columns.str.endswith('A')]
        margin_cols = [col for col in df_county_demo.columns if 'M' in col[4:5]]

        df_county_demo = df_county_demo.drop(columns=['GISJOIN'] + list(cols_A) + margin_cols)
        return df_county_demo

    def county_demo_prep(self):
        df_county_demo = self.get_county_demo()
        f = open('./Data/nhgis0010_ds239_20185_2018_county_codebook.txt', 'r')
        searchlines = f.readlines()
        f.close()
        for i in range(len(searchlines)):
            if '    Margins of error' in searchlines[i]:
                error_margin = i
        col_dict = {}
        for col in df_county_demo.columns[4:]:
            for i, line in enumerate(searchlines):
                # print(i)
                if col[:-4] in line and 'Table' in searchlines[i - 3] and i < error_margin:
                    text = re.sub(r"\s+", " ", searchlines[i - 3].strip())
                    top_col = text.split(':')[1].strip()
                if col in line:
                    text = line.strip().split(':')[1:]
                    col_name = ''.join(map(str, text))
                    col_name = re.sub(r"\s+", " ", col_name).strip()
                    # print(col_name)
                    col_dict[col] = top_col + '_' + col_name
        df_county_demo = self.get_county_demo()
        df_county_demo = df_county_demo.rename(columns=col_dict)
        insurance_cols = [col for col in df_county_demo.columns if 'Insurance' in col]
        hispanic_cols = [col for col in df_county_demo.columns if 'Hispanic' in col]

        df_county_demo = df_county_demo.drop(columns=insurance_cols + hispanic_cols)
        return df_county_demo

    def county_election_data(self):
        county_election = pd.read_csv('./Data/countypres_2000-2016.csv', sep='\t')
        county_election = county_election[county_election.year == 2016]
        maj_party = county_election['party'][
            county_election.groupby(['FIPS'])['candidatevotes'].transform(max) == county_election['candidatevotes']]
        fips = county_election['FIPS'][
            county_election.groupby(['FIPS'])['candidatevotes'].transform(max) == county_election['candidatevotes']]
        fips_party = pd.concat([fips, maj_party], ignore_index=False, axis=1, ).reset_index(drop=True).dropna()
        fips_idx_county = county_election.set_index('FIPS')
        pct_rep = (fips_idx_county[fips_idx_county['party'] == 'republican']['candidatevotes']) / \
                  (fips_idx_county[fips_idx_county['party'] == 'republican']['candidatevotes'] +
                   fips_idx_county[fips_idx_county['party'] == 'democrat']['candidatevotes'])
        pct_rep = pct_rep.reset_index().rename(columns={'candidatevotes': 'pct_republican'})
        fips_party = fips_party.join(pct_rep.set_index('FIPS'), on='FIPS')
        fips_party.FIPS = fips_party.FIPS.astype(int).astype(str).str.zfill(5)
        return fips_party

    def combined_county_data(self):
        df_county_demo = self.county_demo_prep()
        fips_party = self.county_election_data()
        total_cols = [col for col in df_county_demo.columns if 'Total' in col]
        for col in total_cols:
            remaining_cols = [c for c in list(df_county_demo.columns) if c != col]
            for df_col in remaining_cols:
                if col.split('_')[0] in df_col:
                    df_county_demo[df_col] = df_county_demo[df_col] / df_county_demo[col]
        df_county_demo = df_county_demo.join(fips_party.set_index('FIPS'), on='fips')
        df_county_demo = df_county_demo.fillna(df_county_demo.mean())

        return df_county_demo

    def load_Atlas(self,fpath='./Data/Atlas County Info/RuralAtlasData22.xlsx'):
        xls_atlas = pd.ExcelFile(fpath)
        df_atlas = pd.DataFrame()
        for sheet in xls_atlas.sheet_names:
            if sheet != 'Read Me' and sheet != 'Variable Name Lookup':
                df_atlas_sheet = pd.read_excel(xls_atlas, sheet_name=sheet)
                df_atlas_sheet = df_atlas_sheet.drop(columns=['State', 'County'])

                if 'FIPStxt' not in df_atlas_sheet.columns:
                    df_atlas_sheet = df_atlas_sheet.iloc[1:]
                else:
                    df_atlas_sheet = df_atlas_sheet.rename(columns={'FIPStxt': 'FIPS'})

                if len(df_atlas) != 0:
                    df_atlas = df_atlas.join(df_atlas_sheet.set_index('FIPS'), on='FIPS')
                else:
                    df_atlas = df_atlas_sheet.copy()
        df_atlas.FIPS = df_atlas.FIPS.astype(int).astype(str).str.zfill(5)
        df_atlas = df_atlas.fillna(df_atlas.mean())
        return df_atlas

    def load_CUSP(self,sheet_name, excel_fname='./Data/COVID-19 US state policy database (CUSP).xlsx', remove_cols=[]):
        xls = pd.ExcelFile(excel_fname)
        df = pd.read_excel(xls, sheet_name)
        df = df.dropna()
        df = df.drop(columns=['State Abbreviation', 'State FIPS Code'] + remove_cols)
        start_date = datetime.datetime.strptime('2020-03-01', '%Y-%m-%d')
        for col in df.columns:

            if col != 'State':
                if df[col].isin([0, 1]).all() or df[col].isin([0, 1, 2]).all():
                    df[col] = df[col].fillna(df[col].min())
                    continue
                else:
                    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
                    df[col] = df[col].fillna(df[col].max())
                    df['Days_' + col] = (df[col] - start_date).dt.days
                    df = df.drop(columns=[col])
        df = df.rename(columns={'State': 'STATE'})
        return df
