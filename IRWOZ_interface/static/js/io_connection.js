var socket = io.connect('http://' + document.domain + ':' + location.port);

$(document).ready(function() {
    $( '#max_side_domains' ).show();
    $( '#user_side_task_specification' ).hide();
});

function checkmode(element){
  if (element.checked){
     $( '#max_side_domains' ).hide();
     $( '#user_side_task_specification' ).show();
  } else{
    $( '#max_side_domains' ).show();
    $( '#user_side_task_specification' ).hide();
  }
}

function gettzdate(){
    var current_time = new Date().toLocaleTimeString();
    // if ($('#usermode').is(':checked')){
    //   alert("this is a test")
    // }
    return current_time ;
}

function get_area() {
            $.ajax({
              url:'/get_area/',
              data: { "area": $('#area').val().toLowerCase()},
              type:"GET",
              dataType:'json',
                success:function (data) {
                    $.each(data,function(k,v) {
                        if (v == "detected"){
                          document.getElementById("area_res").textContent = "Great, " + $('#area').val().toLowerCase() + " is found."
                        }else if (v == "undetected"){
                            document.getElementById("area_res").textContent = "Sorry, " + $('#area').val().toLowerCase() + " is not found."
                        }else{
                            document.getElementById("area_res").textContent = "Please enter the area first."
                        }
                    })
                }
            })
        };

function get_location() {
            $.ajax({
              url:'/get_location/',
              data: { "area": $('#area').val().toLowerCase(),"location": $('#location').val().toLowerCase()},
              type:"GET",
              dataType:'json',
                success:function (data) {
                    $.each(data,function(k,v) {
                        if (v == "detected"){
                          document.getElementById("location_res").textContent = $('#area').val().toLowerCase() + " and " + $('#location').val().toLowerCase() + " are found."
                        }else if (v == "undetected"){
                            document.getElementById("location_res").textContent = "Area and Loation are not related."
                        }else if (v =="Need Area and Location"){
                            document.getElementById("location_res").textContent = " Area and Location can not be empty."
                        }else if (v =="Need Area"){
                            document.getElementById("location_res").textContent = "Need area for search location."
                        }else if (v =="Need Location"){
                            document.getElementById("location_res").textContent = "Location can not be empty."
                        }
                    })
                }
            })
        };


socket.on( 'connect', function() {
  socket.emit( 'my event', {
    data: 'User Connected'
  } )
  var form = $( 'form' ).on( 'submit', function( e ) {
    e.preventDefault()
    let user_input = $("#message").val();
    let speaker = '';
    if ($('#usermode').is(':checked')){
      speaker = "user"
    } else {
      speaker = "max"
    }
    socket.emit( 'my event', {
      message : user_input,
      speaker : speaker
    } )
  $("#message").val( '' ).focus()
  } )
} )


socket.on('my response', function( msg ) {
  if (msg.message !== undefined) {

    if (msg.message !==""){

      if ($('#usermode').is(':checked')) {

        if (msg.speaker == "max") {
          $("#dialogue").append("<li class=\"clearfix\"><div class=\"message-data\"><img src=\"/static/data/image/max.png\" alt=\"Max\"> " +
              "<span class=\"message-data-time\">" + gettzdate() + "</span></div><div class=\"message other-message\">"
              + msg.message + "</div></li>")
        } else if (msg.speaker == "user") {
          $("#dialogue").append("<li class=\"clearfix\"><div class=\"message-data text-right\"><span class=\"message-data-time\">" + gettzdate() +
              "</span><img src=\"/static/data/image/user.png\" alt=\"User\"></div><div class=\"message other-message float-right\">" + msg.message + "</div></li>")
        }
      } else {
        if (msg.speaker == "user") {
          $("#dialogue").append("<li class=\"clearfix\"><div class=\"message-data\"><img src=\"/static/data/image/user.png\" alt=\"User\"> " +
              "<span class=\"message-data-time\">" + gettzdate() + "</span></div><div class=\"message other-message\">"
              + msg.message + "</div></li>")
        } else if (msg.speaker == "max") {
          $("#dialogue").append("<li class=\"clearfix\"><div class=\"message-data text-right\"><span class=\"message-data-time\">" + gettzdate() +
              "</span><img src=\"/static/data/image/max.png\" alt=\"Max\"></div><div class=\"message other-message float-right\">" + msg.message + "</div></li>")
        }
      }
    } else {
      alert("Please enter the message before you post :)")
    }
  }
})




