

var express = require('express');
var router = express.Router();

const { Client } = require('@elastic/elasticsearch')
const client = new Client({ node: 'http://localhost:9200' })




var test=''
router.get('/', function (req, res){
  res.render('index',{title: 'Express'})

});

router.get('/search', function (req, res){
  res.render('search',{title : 'Search'})
});

router.get('/answer/article' ,function(req,res){
    
  
  
  //data = '這是文章列表，你想看的 tab 是：' + req.query.tab +
  position = req.query.tab
  
  back_word = test[position]["_source"]["BACKGROUND"]
  obje_word = test[position]["_source"]["OBJECTIVES"]
  meth_word = test[position]["_source"]["METHODS"]
  resu_word = test[position]["_source"]["RESULTS"]
  conc_word = test[position]["_source"]["CONCLUSIONS"]
  othe_word = test[position]["_source"]["OTHERS"]

  


  if(typeof(back_word)=="undefined"){
    back_word="0"
  }
  if(typeof(obje_word)=="undefined"){
    obje_word="0"
  }
  if(typeof(meth_word)=="undefined"){
    meth_word="0"
  }
  if(typeof(resu_word)=="undefined"){
    resu_word="0"
  }
  if(typeof(conc_word)=="undefined"){
    conc_word="0"
  }
  if(typeof(othe_word)=="undefined"){
    othe_word="0"
  }


  abstract_word= test[position]["_source"]["Abstract"]
  
  tmp_word=""
  abstract_arr=[]
  for(i=0;i<abstract_word.length;i++){
    
    if(abstract_word[i]=="." && abstract_word[i+1]==" " ){

      tmp_word=tmp_word+'.'
      abstract_arr.push(tmp_word)
      tmp_word=""
      
    }
    else{
      tmp_word=tmp_word+abstract_word[i]
    }
    if(abstract_word[i]=="."  && i==abstract_word.length-1){

      tmp_word=tmp_word+'.'
      abstract_arr.push(tmp_word)
      tmp_word=""
      
    }
  }
  
  abstract_word=""
  
  
  tmp_word=""
  check_list=""
  
  for(i=0;i<abstract_arr.length;i++){{
    clock = 0
    n = back_word.indexOf(abstract_arr[i])
    if(n != -1){
      check_list=check_list+"  &#10003;"
    }
    else{
      check_list=check_list+"  &#10007;"
      clock++
    }
    
    n = obje_word.indexOf(abstract_arr[i])
 
    if(n != -1){
      check_list=check_list+"  &#10003;"
    }
    else{
      check_list=check_list+"  &#10007;"
      clock++
    }

    n = meth_word.indexOf(abstract_arr[i])
    if(n != -1){
      check_list=check_list+"  &#10003;"
    }
    else{
      check_list=check_list+"  &#10007;"
      clock++
    }

    n = resu_word.indexOf(abstract_arr[i])
    if(n != -1){
      check_list=check_list+"  &#10003;"
    }
    else{
      check_list=check_list+"  &#10007;"
      clock++
    }

    n = conc_word.indexOf(abstract_arr[i])
    if(n != -1){
      check_list=check_list+"  &#10003;"
    }
    else{
      check_list=check_list+"  &#10007;"
      clock++
    }

    n = othe_word.indexOf(abstract_arr[i])
    if(n != -1 || clock==5){
      check_list=check_list+"  &#10003; "
    }
    else{
      check_list=check_list+"  &#10007; "
    }

    abstract_word= abstract_word + check_list + abstract_arr[i] + "<br>"
    check_list=""
    
  }}
  console.log(check_list)
  



  /*tmp_word=""
  for(i=0;i<back_word.length;i++){
    if(back_word[i]=='.'){
      n = abstract_word.indexOf(tmp_word)
      abstract_word = abstract_word.slice(0,n) + "<font color=" +"'BLUE'" + ">" +abstract_word.slice(n,n+tmp_word.length) + "</font>" +abstract_word.slice(n+tmp_word.length)
      tmp_word=""
      
    }
    else{
      tmp_word=tmp_word+back_word[i]
    }
  }

  tmp_word=""
  for(i=0;i<obje_word.length;i++){
    if(obje_word[i]=='.'){
      n = abstract_word.indexOf(tmp_word)
      abstract_word = abstract_word.slice(0,n) + "<font color=" +"'BLUE'" +">" +abstract_word.slice(n,n+tmp_word.length) + "</font>" +abstract_word.slice(n+tmp_word.length)
      tmp_word=""
      
    }
    else{
      tmp_word=tmp_word+obje_word[i]
    }
  }
  tmp_word=""
  for(i=0;i<meth_word.length;i++){
    if(meth_word[i]=='.'){
      n = abstract_word.indexOf(tmp_word)
      abstract_word = abstract_word.slice(0,n) + "<font color=" +"'PURPLE'" +">" +abstract_word.slice(n,n+tmp_word.length) + "</font>" +abstract_word.slice(n+tmp_word.length)
      tmp_word=""
      
    }
    else{
      tmp_word=tmp_word+meth_word[i]
    }
  }
  tmp_word=""
  for(i=0;i<resu_word.length;i++){
    if(resu_word[i]=='.'){
      n = abstract_word.indexOf(tmp_word)
      abstract_word = abstract_word.slice(0,n) + "<font color=" +"'ORANGE'" +">" +abstract_word.slice(n,n+tmp_word.length) + "</font>" +abstract_word.slice(n+tmp_word.length)
      tmp_word=""
      
    }
    else{
      tmp_word=tmp_word+resu_word[i]
    }
  }
  tmp_word=""
  for(i=0;i<conc_word.length;i++){
    if(conc_word[i]=='.'){
      n = abstract_word.indexOf(tmp_word)
      abstract_word = abstract_word.slice(0,n) + "<font color=" +"'GREEN'" +">" +abstract_word.slice(n,n+tmp_word.length) + "</font>" +abstract_word.slice(n+tmp_word.length)
      tmp_word=""
      
    }
    else{
      tmp_word=tmp_word+conc_word[i]
    }
  }
  tmp_word=""
  for(i=0;i<othe_word.length;i++){
    if(othe_word[i]=='.'){
      n = abstract_word.indexOf(tmp_word)
      abstract_word = abstract_word.slice(0,n) + "<font color=" +"'BLACK'" +">" +abstract_word.slice(n,n+tmp_word.length) + "</font>" +abstract_word.slice(n+tmp_word.length)
      tmp_word=""
      
    }
    else{
      tmp_word=tmp_word+othe_word[i]
    }
  }
  */
  
  data= 
       "<h2>Title</h2>"
      + "<p>" +"<font size=3 " + ">" + test[position]["_source"]["Title"] +"</font>" + "</p>"
      + "<hr>"
      + "<h2>Authors</h2>"
      + "<p>" +"<font size=3 " + ">" + test[position]["_source"]["Authors"] +"</font>" + "</p>"
      + "<hr>"
      + "<h2>Categories</h2>"
      + "<p>" +"<font size=3 " + ">" + test[position]["_source"]["Categories"] +"</font>" + "</p>"
      + "<hr>"
      + "<h2>Abstract</h2>"
      +  "<p><font size=2 color=" + "red" + ">" +"B=background  " +"</font>" 
      +  "<font size=2 color=" + "blue" + ">" +"J=objectives  " +"</font>"
      +  "<font size=2 color=" + "purple" + ">" +"M=methods  " +"</font>"
      +  "<font size=2 color=" + "orange" + ">" +"R=results  " +"</font>"
      +  "<font size=2 color=" + "green" + ">" +"C=conclusion  " +"</font>"
      +  "<font size=2 color=" + "black" + ">" +"O=others  " +"</font>"
      +   "</p>" 

      +  "<font size=3 color=" + "red" + ">" +"B " +"</font>" 
      +  "<font size=3 color=" + "blue" + ">" +"J " +"</font>"
      +  "<font size=3 color=" + "purple" + ">" +"M " +"</font>"
      +  "<font size=3 color=" + "orange" + ">" +"R " +"</font>"
      +  "<font size=3 color=" + "green" + ">" +"C " +"</font>"
      +  "<font size=3 color=" + "black" + ">" +"O " +"</font>"
      +  "<br>"

      + "<font face=" + "Helvetica" +">"+abstract_word  +"</font>"
      + "<hr>"
 
  
  
 
  res.send(data)
  
 
  
  

  



});






router.post('/search',function(req,res,next){

  
  


  async function run_search () {
    
    
    dsl_bg = { match: { Title : req.body.search }}
    dsl_ob = { match: { Title : req.body.search }}
    dsl_me = { match: { Title : req.body.search }}
    dsl_re = { match: { Title : req.body.search }}
    dsl_co = { match: { Title : req.body.search }}
    dsl_ot = { match: { Title : req.body.search }}



    dsl_source = ["Title","Abstract","Authors","Categories"] 

    choice = req.body.favor
    console.log(choice)
    //問題是一個會出錯
    for(i=0;i<choice.length;i++){
      if(choice[i]=="BACKGROUND"){
        dsl_bg = { match: { BACKGROUND : req.body.search }}
        dsl_source.push("BACKGROUND")
        
      }
      if(choice[i]=="OBJECTIVES"){
        dsl_bg = { match: { OBJECTIVES : req.body.search }}
        dsl_source.push("OBJECTIVES")
      }
      if(choice[i]=="RESULTS"){
        dsl_bg = { match: { RESULTS : req.body.search }}
        dsl_source.push("RESULTS")
      }
      if(choice[i]=="METHODS"){
        dsl_bg = { match: { METHODS : req.body.search }}
        dsl_source.push("METHODS")
      }
      if(choice[i]=="CONCLUSIONS"){
        dsl_bg = { match: { CONCLUSIONS : req.body.search }}
        dsl_source.push("CONCLUSIONS")
      }
      if(choice[i]=="OTHERS"){
        dsl_bg = { match: { OTHERS : req.body.search }}
        dsl_source.push("OTHERS")
      }

    }
    if(choice.length > 6 || choice =="OTHERS"){
      dsl_source.push(choice)
    }
    
  
    
    
    
    const { body } = await client.search({
      

      index: 'engine',
      type: 'type', // uncomment this line if you are using Elasticsearch ≤ 6

      body:{
      
        "size":10,
        "query": {
          "bool": {
            "should": [
              dsl_bg,
              dsl_ob,
              dsl_me,
              dsl_re,
              dsl_co,
              dsl_ot,
              

            ]
          }
        },
        _source : dsl_source
        
      }

    })

    test=body.hits.hits;
    search_word=req.body.search
    res.test;


    
    
    res.redirect('/answer');
    
    

    
  }
  run_search().catch(console.log);
  







});


router.get('/answer',function(req,res,next){
  
  
  change_search = req.query.tab



  async function run_search () {
    
    dsl_source = ["Title","Abstract","Authors","Categories","BACKGROUND","OBJECTIVES","RESULTS","CONCLUSIONS","METHODS","OTHERS"] 

    //問題是一個會出錯

    if(change_search==0){    
      dsl_bg = { match: { BACKGROUND : search_word }}
      dsl_ob = { match: { BACKGROUND : search_word }}
      dsl_me = { match: { BACKGROUND : search_word }}
      dsl_re = { match: { BACKGROUND : search_word }}
      dsl_co = { match: { BACKGROUND : search_word }}
      dsl_ot = { match: { BACKGROUND : search_word }}
      dsl_source.push("BACKGROUND")
      choice="BACKGROUND"
    }
    else if(change_search==1){
      dsl_bg = { match: { OBJECTIVES : search_word }}
      dsl_ob = { match: { OBJECTIVES : search_word }}
      dsl_me = { match: { OBJECTIVES : search_word }}
      dsl_re = { match: { OBJECTIVES : search_word }}
      dsl_co = { match: { OBJECTIVES : search_word }}
      dsl_ot = { match: { OBJECTIVES : search_word }}
      dsl_source.push("OBJECTIVES")
      choice="OBJECTIVES"
    }
    else if(change_search==2){
      dsl_bg = { match: { RESULTS : search_word }}
      dsl_ob = { match: { RESULTS : search_word }}
      dsl_me = { match: { RESULTS : search_word }}
      dsl_re = { match: { RESULTS : search_word }}
      dsl_co = { match: { RESULTS : search_word }}
      dsl_ot = { match: { RESULTS : search_word }}
      dsl_source.push("RESULTS")
      choice="RESULTS"
    }
    else if(change_search==3){
      dsl_bg = { match: { METHODS : search_word }}
      dsl_ob = { match: { METHODS : search_word }}
      dsl_me = { match: { METHODS : search_word }}
      dsl_re = { match: { METHODS : search_word }}
      dsl_co = { match: { METHODS : search_word }}
      dsl_ot = { match: { METHODS : search_word }}
      dsl_source.push("METHODS")
      choice="METHODS"
    }
    else if(change_search==4){
      dsl_bg = { match: { CONCLUSIONS : search_word }}
      dsl_ob = { match: { CONCLUSIONS : search_word }}
      dsl_me = { match: { CONCLUSIONS : search_word }}
      dsl_re = { match: { CONCLUSIONS : search_word }}
      dsl_co = { match: { CONCLUSIONS : search_word }}
      dsl_ot = { match: { CONCLUSIONS : search_word }}
      dsl_source.push("CONCLUSIONS")
      choice="CONCLUSIONS"
    }
    else if(change_search==5){
      dsl_bg = { match: { OTHERS : search_word }}
      dsl_ob = { match: { OTHERS : search_word }}
      dsl_me = { match: { OTHERS : search_word }}
      dsl_re = { match: { OTHERS : search_word }}
      dsl_co = { match: { OTHERS : search_word }}
      dsl_ot = { match: { OTHERS : search_word }}
      dsl_source.push("OTHERS")
      choice="OTHERS"
    }
    else{
      return;
    }

    const { body } = await client.search({
      

      index: 'engine',
      type: 'type', // uncomment this line if you are using Elasticsearch ≤ 6

      body:{
      
        "size":10,
        "query": {
          "bool": {
            "should": [
              dsl_bg,
              dsl_ob,
              dsl_me,
              dsl_re,
              dsl_co,
              dsl_ot,
              

            ]
          }
        },
        _source : dsl_source
        
      }

    })

    test=body.hits.hits;
    res.test;
    
    
    
  }
  run_search().catch(console.log);
  

  
  
  var t = []
  for(i=0;i<test.length;i++){
    t[i] = test[i]["_source"]["Title"]
  }
  

  
  //console.log(test)

  res.render('answer',{
    title1:t[0],
    title2:t[1],
    title3:t[2],
    title4:t[3],
    title5:t[4],
    title6:t[5],
    title7:t[6],
    title8:t[7],
    title9:t[8],
    title10:t[9],
    their_choice:choice
    
  });

  //res.render('answer',{background:b});
  //console.log(test[0]["_index"])
  
  

  
  
  
  


})



module.exports = router;


