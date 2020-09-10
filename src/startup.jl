
using Dates

function startup()

    timenow = now()
    datef   = Dates.format(timenow, "e, dd u yyyy (HH:MM:SS)")

    date    = string(datef)
    host    = gethostname()
    user    = ENV["USER"]

    num_threads = try
        ENV["JULIA_NUM_THREADS"]
    catch e
        if isa(e, KeyError)
            # no JULIA_NUM_THREADS defined; set num_threads to 1
            1
        else
            # unknown error
            error(e)
        end
    end

    println("-------------------------------------------------------------")

    println(raw"                                     _      ")
    println(raw"                                 .:'/   _..._  ")
    println(raw"           Jecco                // ( ```-.._.'  ")
    println(raw"                                \| /    0\___    ")
    println(raw"                                |     0      \    ")
    println(raw"       Julia Einstein           |            /  ")
    println(raw"                                \_       .--'  ")
    println(raw"    Characteristic Code         (_'---'`)      ")
    println(raw"                                / `'---`|  ")
    println(raw"                              ,'        |  ")
    println(raw"              .            .'`          |  ")
    println(raw"              )\       _.-'             ;  ")
    println(raw"             / |    .'`   _            /  ")
    println(raw"           /` /   .'       '.        , |  ")
    println(raw"          /  /   /           \   ;   | |  ")
    println(raw"          |  \  |            |  .|   | |  ")
    println(raw"           \  ``|           /.-' |   | |  ")
    println(raw"            '-..-\       _.;.._  |   |.;-.  ")
    println(raw"                  \    <`.._  )) |  .;-. ))  ")
    println(raw"                  (__.  `  ))-'  \_    ))'   ")
    println(raw"                      `'--``       `````  ")

    println("-------------------------------------------------------------")
    println("")

    println("Run date:          $date")
    println("Run host:          $host")
    println("username:          $user")
    println("")
    println("This process contains $num_threads threads")
    println("")

end
