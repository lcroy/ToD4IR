var socket = io.connect('http://' + document.domain + ':' + location.port);

// obtain time for message
function gettzdate(){
    var current_time = new Date().toLocaleTimeString();
    // if ($('#usermode').is(':checked')){
    //   alert("this is a test")
    // }
    return current_time ;
}

socket.on( 'connect', function() {
    socket.emit( 'my event', {
        data: 'User Connected is',
        flag: 'connection'
    } )
    var form = $( 'form' ).on( 'submit', function( e ) {
        e.preventDefault()
        let utterance = $("#message").val();
        let speaker = "user";
        socket.emit( 'my event', {
            utterance: utterance,
            speaker: speaker,
            flag: 'response'
        } )
        $("#message").val( '' ).focus();
    } )
} )

socket.on('my response', function( msg ) {
  if (msg.message !== undefined) {
        if (msg.message !==""){
         $("#dialogue").append("<li class=\"clearfix\"><div class=\"message-data text-right\"><span class=\"message-data-time\">" + gettzdate() +
              "</span><img src=\"/static/data/image/user.png\" alt=\"User\"></div><div class=\"message other-message float-right\">" + msg.message + "</div></li>");
         $("#dialogue").append("<li class=\"clearfix\"><div class=\"message-data\"><img src=\"/static/data/image/max.png\" alt=\"Max\"> " +
              "<span class=\"message-data-time\">" + gettzdate() + "</span></div><div class=\"message other-message\">"
              + msg.tod_res + "</div></li>");

      //   assign belief state
            document.getElementById("belief_state").textContent = msg.bf;
      //   assign system actions
            document.getElementById("sys_act").textContent = msg.sys;
      //   assign task related response
            document.getElementById("tt_res").textContent = msg.tt;
      //   assign small talk related response
            document.getElementById("st_res").textContent = msg.st;
      } else {
      alert("Please enter the message before you post :)")
    }
  }
})




