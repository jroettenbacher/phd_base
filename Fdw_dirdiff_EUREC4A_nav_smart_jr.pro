;cd //projekt_agmwend/data/Cirrus_HL/00_Tools/03_SMART/01_simulations_Fdw/
;.r Fdw_dirdiff_EUREC4A_nav_smart.pro

;**********************************************************************
;Use this to perform the fdw simulations using the iMAR Navigation Data
;**********************************************************************



pro sod2hhmmssarray,sod_input,hh,mm,ss
 sod = double(sod_input)
 hh = fix(sod/3600d)
 mmf = (sod/3600d - double(hh) ) * 60d
 mm = fix(mmf)
 ssf = ( mmf - float(mm) ) * 60.0
 ss = round(ssf)
 so = where(ss ge 60,mso)
 if mso gt 0 then begin
    ss[so]-=60
    mm[so]++
    endif
 mo=where(mm ge 60,mmo)
 if mmo gt 0 then begin
       mm[mo]-=60
       hh[mo]++
       endif
 end
function start_libradtran2,ts,lati,longi,alti,atmf,ozone,date
  lib_file_path = '/projekt_agmwend/data/Cirrus_HL/00_Tools/03_SMART/01_simulations_Fdw/libradtran/'
  openw,1,lib_file_path+'dd'+date+'.inp'
    printf,1,'data_files_path /opt/libradtran/2.0.4/share/libRadtran/data            # location of internal libRadtran data'
    printf,1,'rte_solver disort'
    printf,1,'mol_abs_param lowtran'
    printf,1,'atmosphere_file /projekt_agmwend/data/Cirrus_HL/00_Tools/03_SMART/01_simulations_Fdw/add_data/afglt_evi.dat'
;    printf,1,'radiosonde '+atmf+' H2O RH'
    printf,1,'time '+ts
    if float(lati) lt 0.0 then printf,1,'latitude S '+string(abs(float(lati)))
    if float(lati) ge 0.0 then printf,1,'latitude N '+string(abs(float(lati)))
    if float(longi) lt 0.0 then printf,1,'longitude W '+string(abs(float(longi)))
    if float(longi) ge 0.0 then printf,1,'longitude E '+string(abs(float(longi)))
    if alti lt 0.0 then printf,1,'zout 0 #'+alti
    if alti gt 0.0 then printf,1,'zout '+alti
    printf,1,'wavelength 250 2500'
    printf,1,'source solar  /projekt_agmwend/data/Cirrus_HL/00_Tools/03_SMART/01_simulations_Fdw/add_data/NewGuey2003_AlbedoWavelengths.dat'
    printf,1,'output_user lambda sza edir edn'
    printf,1,'albedo_file /projekt_agmwend/data/Cirrus_HL/00_Tools/03_SMART/01_simulations_Fdw/add_data/kentucky_blue_grass.dat'
    printf,1,'mol_modify O3 '+ozone + '  DU'
  close,1
  cmd = 'uvspec < '+lib_file_path + 'dd'+date+'.inp > '+lib_file_path+'dd'+date+'.out'
  spawn,cmd,spawnresult,spawnerror
  return,spawnerror
end

close,/all





;=======================================================================================================================================
;set time interval for simulations
;=======================================================================================================================================

time_step = 1800L;2L;240L;120L;120L																		;*********** SET TIME Interval here **************



;=======================================================================================================================================
;select day / days to simulate
;=======================================================================================================================================





;date='20200115a'
;date='20200118a'
;date='20200119a'
;date='20200122a'
;date='20200124a'
;date='20200126a'
;date='20200128a'
;date='20200130a'
;date='20200131a'
;date='20200202a'
;date='20200205a'
date='20200207a'
;date='20200209a'
;date='20200211a'
;date='20200213a'
;date='20200215a'



path1 = '/projekt_agmwend/data_raw/Cirrus_HL_raw_only/01_Flights/Flight_'+date
lib_file_path = '/projekt_agmwend/data/Cirrus_HL/00_Tools/03_SMART/01_simulations_Fdw/libradtran/'

navfile = file_search(path1+'/horidata/NavCommand/','Nav_GPSPos*')

print,'reading nav file: '+navfile

flightname = date
case date of  ; OMI estimates from http://es-ee.tor.ec.gc.ca/e/ozone/Curr_allmap_g.htm or http://www.temis.nl/protocols/O3global.html
 '20200115a': o3 =  375.
 '20200118a': o3 =  375.
 '20200119a': o3 =  300.
 '20200122a': o3 =  375.
 '20200124a': o3 =  375.
 '20200126a': o3 =  375.
 '20200128a': o3 =  375.
 '20200130a': o3 =  375.
 '20200131a': o3 =  375.
 '20200202a': o3 =  375.
 '20200205a': o3 =  375.
 '20200207a': o3 =  375.
 '20200209a': o3 =  375.
 '20200211a': o3 =  375.
 '20200213a': o3 =  375.
 '20200215a': o3 =  375.
endcase

;==============================================================
;read SMART NAV file
;==============================================================
print,'read SMART NAV'  & wait,0.1

openr,1,navfile
dummy=''
for i=0,12 do readf,1,dummy

data=fltarr(8,file_lines(navfile)-13)
readf,1,data

close,1


sod=data(1,*)  ;*3600.
lat=data(3,*)
lon=data(2,*)
alt=data(4,*)/1000.

print,'** Data is read, starting day '+date

case date of
 '20200115a': atmosfile = '/projekt_agmwend/data/EUREC4A/03_Soundings/RS_for_libradtran/Muenchen_Oberschleissheim_10868/'+strmid(date,4,4)+'_12.dat'
 '20200118a': atmosfile = '/projekt_agmwend/data/EUREC4A/03_Soundings/RS_for_libradtran/Muenchen_Oberschleissheim_10868/'+strmid(date,4,4)+'_12.dat'
 '20200119a': atmosfile = '/projekt_agmwend/data/EUREC4A/03_Soundings/RS_for_libradtran/Muenchen_Oberschleissheim_10868/'+strmid(date,4,4)+'_12.dat'
 else:        atmosfile = '/projekt_agmwend/data/Cirrus_HL/02_Soundings/RS_for_libradtran/Meiningen_10548/'+strmid(date,4,4)+'_12.dat'
endcase

print,'reading atmosphere file: '+atmosfile


;sod2hhmmssarray,sod,h,m,s
datestring = strmid(date,0,4) + ' ' + strmid(date,4,2) + ' ' + strmid(date,6,2) + ' '




;start loop for simulations
 j=0L
 counter = 0L
 jindex = 0L
 check = long(sod[j])
 while j lt n_elements(sod) do begin
; while j lt 60. do begin
    if (j eq 0L) or (j eq n_elements(sod)-1) or (long(sod[j]) ge time_step + check) then begin
       if (j eq n_elements(sod)-1) then s[j]++  ; make sure last second is fully covered

       timestring = datestring + string(h[j],format='(I2.2)') + ' ' + string(m[j],format='(I2.2)') + ' ' + string(s[j],format='(I2.2)')

       waste = start_libradtran2(timestring,string(lat[j]),string(lon[j]),string(alt[j]),atmosfile,string(o3),date)
       ofile=lib_file_path+'dd'+date+'.out'
       if (j eq 0) then begin
           nwl=file_lines(ofile)
           print,file_lines(ofile)
           dd0 = fltarr(4,nwl)
        endif
        openr,2,ofile
          readf,2,dd0
        close,2
        Fdwdummy = (dd0[2,*]+dd0[3,*])/1000. 			; clear sky irradiance
        ddummy = dd0[2,*] / (dd0[2,*]+dd0[3,*]) ; direct fraction
        if (j eq 0) then begin
          wl = reform(dd0[0,*])
          dd = ddummy
          Fdw = Fdwdummy
        endif else begin
          dd = [dd, ddummy]
          Fdw = [Fdw, Fdwdummy]
        endelse
       jindex = [jindex, j]
       counter++
       check = long(sod[j])
    endif
    if (long(sod[j]) gt time_step + check)  then check = long(sod[j])
    j++
 print,j,n_elements(sod)
 endwhile
 jindex=jindex[1:*]
 toohigh = where(dd gt 1.0,nh)
 if nh gt 0 then dd[toohigh]=-7.
 toolow = where(dd lt 0.0,nl)
 if nl gt 0 then dd[toolow]=-8.
 baddata = where(finite(dd) ne 1,nb)
 if nb gt 0 then dd[baddata]=-9.

 ; Finished working on this day; write results to file

 ; write direct fraction
 print,'Data output for '+flightname
 outfile = '/projekt_agmwend/data/Cirrus_HL/01_Flights/Flight_'+date+'/SMART/DirectFraction_'+date+'_R0.dat'
 openw,oo,outfile,/GET_LUN

   printf,oo,'36 1001'  ; number of total header lines, then '1001'
   printf,oo,'Wendisch, Manfred' ; PI
   printf,oo,'LIM, Leipzig University' ; Affil
   printf,oo,'Direct fraction of downward irradiance along flight track, calculated by libRadtran 2.0.4 using iMAR nav data'
   printf,oo,'Cirrus-HL Flight '+date
   printf,oo,'1 1'  ; only one file
case strmid(systime(/UTC),4,3) of
 'Jan': sysmonth='01'
 'Feb': sysmonth='02'
 'Mar': sysmonth='03'
 'Apr': sysmonth='04'
 'May': sysmonth='05'
 'Jun': sysmonth='06'
 'Jul': sysmonth='07'
 'Aug': sysmonth='08'
 'Sep': sysmonth='09'
 'Oct': sysmonth='10'
 'Nov': sysmonth='11'
 'Dec': sysmonth='12'
 ELSE: sysmonth='systime_error'
endcase
   printf,oo,'2021 '+strmid(date,4,2)+' '+strmid(date,6,2)+' '+strmid(systime(/UTC),20,4)+' '+sysmonth+' '+strmid(systime(/UTC),8,2)
   printf,oo,'0' ; data interval in seconds, 0 = n/a
   printf,oo,'sod (seconds of day; number of seconds from 0000 UTC)'
   printf,oo,strtrim(string(nwl),2) ; number of data columns, not counting time
   printf,oo,'1' ; Scale factor
   printf,oo,'-9' ; Missing data value
   printf,oo,'Latitude, in degrees, north positive'
   printf,oo,'Longitude, in degrees, east positive'
   printf,oo,'Direct fraction of downward irradiance at '+strtrim(string(nwl),2)+' wavelengths'
   printf,oo,'1' ; number of special comment lines following
   printf,oo,'Special comments: None.'
   printf,oo,'18'  ;number of comment lines following
   printf,oo,'PI_CONTACT_INFO: Address: LIM, Stephanstr.3, 04103 Leipzig, Germany. Email: m.wendisch@uni-leipzig.de'
   printf,oo,'PLATFORM: HALO research aircraft, D-ADLR'
   printf,oo,'LOCATION: given in data'
   printf,oo,'ASSOCIATED_DATA: none'
   printf,oo,'INSTRUMENT_INFO: Model data'
   printf,oo,'DATA_INFO: Dimensionless fractions.'
   printf,oo,'UNCERTAINTY: 1% (much more in case of overlying clouds which are not represented in the model)'
   printf,oo,'ULOD_FLAG: -7'
   printf,oo,'ULOD_VALUE: 1.0'
   printf,oo,'LLOD_FLAG: -8'
   printf,oo,'LLOD_VALUE: 0.0'
   printf,oo,'DM_CONTACT_INFO: Andr� Ehrlich, a.ehrlich@uni-leipzig.de'
   printf,oo,'PROJECT_INFO: EUREC4A Campaign, 18 Jan - 18 Feb, Bridgetown, Barbados. http://www.eurec4a.eu'
   printf,oo,'STIPULATIONS_ON_USE: Use of these data requires prior OK from the PI. The EUREC4A Data Protocol applies.'
   printf,oo,'OTHER_COMMENTS: none'
   printf,oo,'REVISION: R0'
   printf,oo,'R0: no comments'
   datatitle='sod         Lat(N)       Lon(E)'
   for dtixxx=1,nwl-1 do datatitle+=string(wl[dtixxx],format='(F15.2)')
   printf,oo,datatitle
   for k=0L, counter-1 do begin
       j=jindex[k]
       printf,oo,sod[j],lat[j],lon[j],dd[k,1:*],format='(I5.5,2F13.6,'+strtrim(string(nwl-1),2)+'F15.7)'
   endfor ; k
   close,oo & free_lun,oo

 ; write Fwd clear sky
 print,'Data output for '+flightname
 outfile = '/projekt_agmwend/data/Cirrus_HL/01_Flights/Flight_'+date+'/SMART/Fdn_clear_sky_'+date+'_R0.dat'
 openw,oo,outfile,/GET_LUN

   printf,oo,'36 1001'  ; number of total header lines, then '1001'
   printf,oo,'Wendisch, Manfred' ; PI
   printf,oo,'LIM, Leipzig University' ; Affil
   printf,oo,'Clear sky downward irradiance along flight track, calculated by libRadtran 2.0.4 using iMAR nav data'
   printf,oo,'Cirrus_HL Flight '+date
   printf,oo,'1 1'  ; only one file
case strmid(systime(/UTC),4,3) of
 'Jan': sysmonth='01'
 'Feb': sysmonth='02'
 'Mar': sysmonth='03'
 'Apr': sysmonth='04'
 'May': sysmonth='05'
 'Jun': sysmonth='06'
 'Jul': sysmonth='07'
 'Aug': sysmonth='08'
 'Sep': sysmonth='09'
 'Oct': sysmonth='10'
 'Nov': sysmonth='11'
 'Dec': sysmonth='12'
 ELSE: sysmonth='systime_error'
endcase
   printf,oo,'2021 '+strmid(date,4,2)+' '+strmid(date,6,2)+' '+strmid(systime(/UTC),20,4)+' '+sysmonth+' '+strmid(systime(/UTC),8,2)
   printf,oo,'0' ; data interval in seconds, 0 = n/a
   printf,oo,'sod (seconds of day; number of seconds from 0000 UTC)'
   printf,oo,strtrim(string(nwl),2) ; number of data columns, not counting time
   printf,oo,'1' ; Scale factor
   printf,oo,'-9' ; Missing data value
   printf,oo,'Latitude, in degrees, north positive'
   printf,oo,'Longitude, in degrees, east positive'
   printf,oo,'Clear sky downward irradiance at '+strtrim(string(nwl),2)+' wavelengths'
   printf,oo,'1' ; number of special comment lines following
   printf,oo,'Special comments: None.'
   printf,oo,'18'  ;number of comment lines following
   printf,oo,'PI_CONTACT_INFO: Address: LIM, Stephanstr.3, 04103 Leipzig, Germany. Email: m.wendisch@uni-leipzig.de'
   printf,oo,'PLATFORM: HALO research aircraft, D-ADLR'
   printf,oo,'LOCATION: given in data'
   printf,oo,'ASSOCIATED_DATA: none'
   printf,oo,'INSTRUMENT_INFO: Model data'
   printf,oo,'DATA_INFO: Irradiance in W m-2 nm-1.'
   printf,oo,'UNCERTAINTY: 1% (little more in case of bright surfaces as kentucky_blue_grass is assumed in the model)'
   printf,oo,'ULOD_FLAG: -7'
   printf,oo,'ULOD_VALUE: 1.0'
   printf,oo,'LLOD_FLAG: -8'
   printf,oo,'LLOD_VALUE: 0.0'
   printf,oo,'DM_CONTACT_INFO: Andr� Ehrlich, a.ehrlich@uni-leipzig.de'
   printf,oo,'PROJECT_INFO: EUREC4A Campaign, 18 Jan - 18 Feb, Bridgetown, Barbados. http://www.eurec4a.eu'
   printf,oo,'STIPULATIONS_ON_USE: Use of these data requires prior OK from the PI. The EUREC4A Data Protocol applies.'
   printf,oo,'OTHER_COMMENTS: none'
   printf,oo,'REVISION: R0'
   printf,oo,'R0: no comments'
   datatitle='sod         Lat(N)       Lon(E)'
   for dtixxx=1,nwl-1 do datatitle+=string(wl[dtixxx],format='(F15.2)')
   printf,oo,datatitle
   for k=0L, counter-1 do begin
       j=jindex[k]
       printf,oo,sod[j],lat[j],lon[j],Fdw[k,1:*],format='(I5.5,2F13.6,'+strtrim(string(nwl-1),2)+'F15.7)'
   endfor ; k
   close,oo & free_lun,oo



end
