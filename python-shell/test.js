var PythonShell = require('python-shell');

// Creates pyshell instance
var pyshell = new PythonShell('echo_text.py', {
    mode: 'text'
});
// Attaches listen for 'data'
pyshell.stdout.on('data', function (data) {
    console.log('TEXT: ' + data);
});
// Sends information
pyshell.send('hello').send('world')
// Ends shell, commenting this out causes the text pyshell to not work
//pyshell.end(function (err) {
//    if (err) return console.log(err);
//});

// Same things, but with JSON
var pyshell = new PythonShell('echo_json.py', {
    mode: 'json'
});
pyshell.stdout.on('data', function (data) {
    console.log('JSON: ' + data);
});
pyshell.send({key: 'value'}).send(['a','r','r','a','y']).end(function (err) {
    if (err) return console.log(err);
});
