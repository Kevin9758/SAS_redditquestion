options validvarname = any; 

/* Example data where we want to find the weighted frequency of
   the sum of values = 't' in each column. */
   
  data nt(drop=id);
  input id x1 $ x2 $ x3 $ x4 $ x5 $;
  datalines;
  1 f t f f f
  2 t t t f t
  3 t t t t t
  4 f f f f f
  5 t f t t f
  ;
run;

* n = number of X's / columns;
%let n = 5;


* Transpose the dataset in order to compute row sums with array;

proc transpose data=nt out = tn;
  var x1-x&n;
run;


data rowsum(keep=_name_ newsum);
  set tn;
  array rs[*]col1-col5;
  do i = 1 to &n;
    if rs[i] = 't' then sum+1;
  end;

  newsum = sum - lag(sum);
    if _name_ = 'x1' then newsum = sum;
run;

proc freq data= rowsum;
  weight newsum;
  table _name_/ out = results outcum;
run;


proc print data=results;
run;

data final_table(rename= (_name_ = X));
  set results;
  'n (% of total)'n = cat(count,' ','(' , round(cum_pct,0.1),'%', ')');
  drop percent count cum_freq cum_pct; 
run;


