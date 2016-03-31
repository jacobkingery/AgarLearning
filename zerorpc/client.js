var zerorpc = require("zerorpc");

var client = new zerorpc.Client();
client.connect("tcp://127.0.0.1:4242");

client.invoke("hello", "RPC", function(error, res, more) {
    console.log('Received ' + typeof res + ' ' + res);
});

client.invoke("hello", {a: 'b'}, function(error, res, more) {
    console.log('Received ' + typeof res + ' ' + res);
    console.log(res.a);
});

client.invoke("hello", [1,2,3], function(error, res, more) {
    console.log('Received ' + typeof res + ' ' + res);
    console.log(res[1]);
});
